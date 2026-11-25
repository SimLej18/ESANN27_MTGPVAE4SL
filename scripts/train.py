"""Training of one of the three models (mtgpvae.prior.MODELS) on LSFB100, from the pre-trained VAE.

- Init: the pre-trained VAE (--vae) and the default prior HPs (prior.Prior.init); or, to fine-tune, a full trained
  model (--init: VAE and prior HPs), with a fresh optimiser.
- Adam on everything (VAE and prior HPs), learning rate with a cosine decay over --decay-steps (default: --steps),
  gradient clipping 1.
- Batch: N_SIGNS signs x N_CLIPS clips (both drawn without replacement), loss of mtgpvae.model.terms.

Paper runs (`just train-all`): the three models, 25k steps; then the MTGP-VAE fine-tuned with a tempered CE
(--init, --beta 5e-4, --ce-temp 10, --lr 5e-4), 5k steps of a cosine decay set for 15k.

Writes {out}/{name}.{eqx,json} (history: mean loss terms of every CHUNK steps).
Usage: python scripts/train.py --model mtgp_vae [--steps 25000] [--out checkpoints]
"""
import os

# Triton GEMM gives NaN / wrong values on GPU for these models under value_and_grad
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
import jax

jax.config.update("jax_enable_x64", True)

import argparse
import time

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax

from mtgpvae.data import load as load_data
from mtgpvae.model import N_CLIPS, N_SIGNS, TERMS, Model, save, terms
from mtgpvae.model import load as load_model
from mtgpvae.prior import MODELS, Prior
from mtgpvae.vae import BETA
from mtgpvae.vae import load as load_vae

LR, CLIP_NORM = 1e-4, 1.0
CHUNK = 500  # steps per jitted scan


def sign_table(signs):
    """Clip indices of each sign (n_signs, max count), rows padded by repeating the sign's own clips; counts."""
    counts = np.bincount(signs)
    by_sign = np.zeros((len(counts), max(counts.max(), N_CLIPS)), int)
    for s in range(len(counts)):
        by_sign[s] = np.resize(np.flatnonzero(signs == s), by_sign.shape[1])
    return by_sign, counts


def draw(key, by_sign, counts):
    """N_SIGNS signs and N_CLIPS clips of each, without replacement: clip indices (N_SIGNS, N_CLIPS)."""
    k_sign, k_clip = jax.random.split(key)
    signs = jax.random.choice(k_sign, len(counts), (N_SIGNS,), replace=False)
    u = jax.random.uniform(k_clip, (N_SIGNS, by_sign.shape[1]))
    j = jnp.argsort(jnp.where(jnp.arange(by_sign.shape[1]) < counts[signs][:, None], u, 1.0 + u), -1)[:, :N_CLIPS]
    return by_sign[signs[:, None], j]


@eqx.filter_jit
def train_chunk(model, opt_state, x, idx, lengths, by_sign, counts, key, opt, beta, ce_temp):
    """CHUNK steps; returns the mean loss and loss terms over the chunk."""
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def step(carry, key):
        params, opt_state = carry
        k_draw, k_loss = jax.random.split(key)
        j = draw(k_draw, by_sign, counts)
        (value, t), grads = eqx.filter_value_and_grad(
            lambda p: terms(eqx.combine(p, static), x[j], idx[j], lengths[j], k_loss, beta, ce_temp), has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return (eqx.apply_updates(params, updates), opt_state), (value, t)

    (params, opt_state), (value, t) = jax.lax.scan(step, (params, opt_state), jax.random.split(key, CHUNK))
    return eqx.combine(params, static), opt_state, jnp.concatenate([value.mean()[None], t.mean(0)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--vae", default="checkpoints/vae_pretrained", help="pre-trained VAE, no extension")
    parser.add_argument("--init", default=None, help="fine-tuning: full model to start from, no extension")
    parser.add_argument("--steps", type=int, default=25_000)
    parser.add_argument("--decay-steps", type=int, default=None, help="length of the cosine decay (default: --steps)")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--beta", type=float, default=BETA, help="weight of the KL and the CE (model.terms)")
    parser.add_argument("--ce-temp", type=float, default=1.0, help="CE temperature (model.terms)")
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--name", default=None, help="default: --model")
    args = parser.parse_args()
    if args.steps % CHUNK:
        raise SystemExit(f"--steps must be a multiple of {CHUNK}")
    name = args.name or args.model

    tr = load_data("training")
    by_sign, counts = sign_table(tr["signs"])
    print(f"{len(counts)} signs, {len(tr['signs'])} training clips, {tr['lengths'].sum()} frames", flush=True)
    if args.init:
        model, _ = load_model(args.init)
        if (model.prior.sign_kind, model.prior.clip_kind) != MODELS[args.model]:
            raise SystemExit(f"--init is not a {args.model} model")
    else:
        vae, _ = load_vae(args.vae)
        model = Model(vae, Prior.init(sum(vae.dims), *MODELS[args.model]))
    opt = optax.chain(optax.clip_by_global_norm(CLIP_NORM),
                      optax.adam(optax.cosine_decay_schedule(args.lr, args.decay_steps or args.steps)))
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    meta = {"model": args.model, "vae": None if args.init else args.vae, "init": args.init, "steps": args.steps,
            "decay_steps": args.decay_steps or args.steps, "lr": args.lr, "beta": args.beta, "ce_temp": args.ce_temp,
            "dims": list(model.vae.dims), "batch": [N_SIGNS, N_CLIPS]}
    x, idx, lengths = (jax.device_put(tr[k]) for k in ("x", "idx", "lengths"))
    by_sign, counts = jnp.asarray(by_sign), jnp.asarray(counts)
    os.makedirs(args.out, exist_ok=True)
    history, t0 = [], time.time()
    for i, k in enumerate(jax.random.split(jax.random.key(0), args.steps // CHUNK)):
        model, opt_state, st = train_chunk(model, opt_state, x, idx, lengths, by_sign, counts, k, opt, args.beta,
                                           args.ce_temp)
        st, done = np.asarray(st), (i + 1) * CHUNK
        history.append(st.tolist())
        if not np.all(np.isfinite(st)):
            raise SystemExit(f"non-finite training stats at {done} steps: {st}")
        print(f"  {done / 1000:g}k, {time.time() - t0:.0f}s, "
              + " ".join(f"{t} {v:.4g}" for t, v in zip(["loss"] + TERMS, st)), flush=True)
    save(f"{args.out}/{name}", model, {**meta, "time": time.time() - t0, "history": history})
    print(f"saved {args.out}/{name}", flush=True)


if __name__ == "__main__":
    main()
