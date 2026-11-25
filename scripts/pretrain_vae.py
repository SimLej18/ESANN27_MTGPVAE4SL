"""Pre-training of the body-part VAE (mtgpvae/vae.py): frame-level, N(0, I) prior, on LSFB-CONT frames (`just data-cont`).

Batch 1024, Adam 1e-3 with cosine decay, gradient clipping 1, 1M steps (~30 min on one GPU).
At the end, reconstruction errors (z = mu) on 20k LSFB100 validation frames, in shoulder widths, and active latent
dims per part (mean KL > 0.1).

Writes {out}/vae_pretrained.{eqx,json}.
Usage: python scripts/pretrain_vae.py [--steps 1000000] [--out checkpoints]
"""
import argparse
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mtgpvae.data import load, load_cont
from mtgpvae.pose import N_LANDMARKS, SLICES
from mtgpvae.vae import DIMS, PARTS, PartVAE, loss, part_means, save

BATCH, LR, CLIP_NORM = 1024, 1e-3, 1.0
CHUNK = 1_000  # steps per jitted scan
N_EVAL = 20_000
GROUPS = {"body": ["upper_pose"], "hands": ["left_hand", "right_hand"], "face": ["lips", "left_eye", "right_eye"]}


@eqx.filter_jit
def train_chunk(model, opt_state, data, key, opt):
    """CHUNK steps; returns the chunk's mean loss, mean rec, mean kl and max gradient norm."""
    params, static = eqx.partition(model, eqx.is_array)
    k_idx, k_steps = jax.random.split(key)
    batches = data[jax.random.randint(k_idx, (CHUNK, BATCH), 0, len(data))]

    def batch_loss(p, x, k):
        losses, (rec, kl) = jax.vmap(loss, in_axes=(None, 0, 0))(eqx.combine(p, static), x, jax.random.split(k, len(x)))
        return losses.mean(), (rec.mean(), kl.mean())

    def step(carry, inputs):
        params, opt_state = carry
        (value, aux), grads = eqx.filter_value_and_grad(batch_loss, has_aux=True)(params, *inputs)
        updates, opt_state = opt.update(grads, opt_state, params)
        return (eqx.apply_updates(params, updates), opt_state), (value, *aux, optax.global_norm(grads))

    (params, opt_state), (value, rec, kl, gnorm) = jax.lax.scan(
        step, (params, opt_state), (batches, jax.random.split(k_steps, CHUNK)))
    return eqx.combine(params, static), opt_state, jnp.stack([value.mean(), rec.mean(), kl.mean(), gnorm.max()])


def evaluate(model, x):
    """Mean landmark errors of the reconstructions (z = mu) of frames x (n, 274), and active dims per part."""
    mu, logvar = jax.jit(jax.vmap(model.encode))(x)
    rec = np.asarray(jax.jit(jax.vmap(model.decode))(mu)).reshape(len(x), N_LANDMARKS, 2)
    x = np.asarray(x).reshape(len(x), N_LANDMARKS, 2)
    dist = np.linalg.norm(rec - x, axis=-1)
    out = {g: float(np.mean([dist[:, SLICES[p]].mean() for p in ps])) for g, ps in GROUPS.items()}
    for h in ("left_hand", "right_hand"):
        s = SLICES[h]
        out[f"{h}_wrist"] = float(dist[:, s.start].mean())
        shape = lambda a: a[:, s.start + 1:s.stop] - a[:, s.start, None]
        out[f"{h}_shape"] = float(np.linalg.norm(shape(rec) - shape(x), axis=-1).mean())
    kl = np.asarray(0.5 * (mu**2 + jnp.exp(logvar) - 1 - logvar).mean(0))
    out["active_per_part"] = {p: int((k > 0.1).sum()) for p, k in zip(PARTS, np.split(kl, np.cumsum(model.dims)[:-1]))}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()
    if args.steps % CHUNK:
        raise SystemExit(f"--steps must be a multiple of {CHUNK}")

    frames = load_cont()
    print(f"training frames: {frames.shape}", flush=True)
    data = jax.device_put(frames)
    model = PartVAE(DIMS, part_means(frames), jax.random.key(0))
    opt = optax.chain(optax.clip_by_global_norm(CLIP_NORM), optax.adam(optax.cosine_decay_schedule(LR, args.steps)))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    val = load("validation")
    x_eval = val["x"][np.arange(val["x"].shape[1]) < val["lengths"][:, None]]  # real frames only
    x_eval = jnp.asarray(x_eval[np.random.default_rng(1).choice(len(x_eval), N_EVAL, replace=False)])

    os.makedirs(args.out, exist_ok=True)
    meta = {"dims": list(DIMS), "steps": args.steps, "batch": BATCH, "lr": LR}
    history, t0 = [], time.time()
    for i, k in enumerate(jax.random.split(jax.random.key(1), args.steps // CHUNK)):
        model, opt_state, stats = train_chunk(model, opt_state, data, k, opt)
        stats, done = np.asarray(stats), (i + 1) * CHUNK
        history.append(stats.tolist())
        if not np.all(np.isfinite(stats)):
            raise SystemExit(f"non-finite training stats at {done} steps: {stats}")
        if i == 0 or done % 50_000 == 0:
            print(f"  {done // 1000}k steps, {time.time() - t0:.0f}s, loss {stats[0]:.4f} (rec {stats[1]:.4f}, "
                  f"kl {stats[2]:.1f}, max grad norm {stats[3]:.2f})", flush=True)
    res = evaluate(model, x_eval)
    save(f"{args.out}/vae_pretrained", model, {**meta, "time": time.time() - t0, "history": history, "eval": res})
    print(f"saved {args.out}/vae_pretrained: {res}", flush=True)


if __name__ == "__main__":
    main()
