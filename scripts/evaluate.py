"""Sign classification of a trained model (Table 1 of the paper), on LSFB100 testing (or validation).

A clip is classified by its codes' likelihood under every sign (mtgpvae.inference), each sign's q(mu_s) coming from all
its training clips. Validation / testing clips flagged left-dominant are mirrored first (as the training clips).
- default: under the model's own prior;
- --posthoc: under a Magma prior (traj, gp) fitted by maximum likelihood on the model's frozen training codes (LBFGS,
  at most 500 steps, ~1 h on a GPU), saved as {ckpt}_posthoc.eqx and reused if present.
Metrics: top-1 / top-5 accuracy, balanced (mean per-sign recall) and not. Higher is better.

Writes {out}/{name}_{split}.json, name = the checkpoint name (+ "_posthoc").
Usage: python scripts/evaluate.py checkpoints/mtgp_vae [--posthoc] [--split testing] [--out results]
       python scripts/evaluate.py --table [--out results]   (Table 1 from the json files)
"""
import jax

jax.config.update("jax_enable_x64", True)

import argparse
import json
import os
import time

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from mtgpvae.data import load as load_data, signs as sign_names
from mtgpvae.inference import class_ll, encode, fit_magma, grid_codes, sign_means, sign_stats
from mtgpvae.model import chols, load as load_model
from mtgpvae.pose import mirror
from mtgpvae.prior import Prior

TABLE = [("Sign-VAE", "sign_vae"), ("Sign-GP-VAE", "sign_gp_vae"), ("Sign-GP-VAE + MTGP post-hoc", "sign_gp_vae_posthoc"),
         ("MTGP-VAE", "mtgp_vae")]


def evaluate(ckpt, posthoc, split, out):
    name = os.path.basename(ckpt) + ("_posthoc" if posthoc else "")
    model, _ = load_model(ckpt)
    vae, prior = model.vae, model.prior
    n_signs = len(sign_names())
    tr = load_data("training")
    mu_tr, lv_tr = encode(vae, tr["x"])
    if posthoc:
        path = f"{ckpt}_posthoc.eqx"
        if os.path.exists(path):
            prior = eqx.tree_deserialise_leaves(path, Prior.init(mu_tr.shape[-1], "traj", "gp"))
        else:
            m0 = np.asarray(prior.m())
            y_tr, lam_tr = grid_codes(m0, mu_tr, lv_tr, tr["idx"])
            t0 = time.time()
            prior, nll, n_steps = fit_magma(y_tr + m0[:, None] * (lam_tr > 0), lam_tr, tr["signs"], n_signs)
            eqx.tree_serialise_leaves(path, prior)
            print(f"post-hoc Magma fitted (nll {nll:.4f}, {n_steps} LBFGS steps), {time.time() - t0:.0f}s", flush=True)
    m0 = np.asarray(prior.m())
    l0, ls, k_clip = (np.asarray(a) for a in chols(prior))
    y_tr, lam_tr = grid_codes(m0, mu_tr, lv_tr, tr["idx"])
    stats = sign_stats(ls, y_tr, lam_tr, tr["signs"], n_signs)
    means = sign_means(jnp.asarray(l0), jnp.asarray(stats[0]), jnp.asarray(stats[1]))

    q = load_data(split)
    x = np.where(q["mirror"][:, None, None], mirror(q["x"]), q["x"])
    y, lam = grid_codes(m0, *encode(vae, x), q["idx"])
    ll = class_ll(k_clip, means, y, lam)
    true = q["signs"]
    rank = (ll > ll[np.arange(len(true)), true][:, None]).sum(1)  # signs strictly more likely than the true one
    present = np.unique(true)
    recall = {k: np.array([np.mean(rank[true == s] < k) for s in present]) for k in (1, 5)}
    res = {"ckpt": ckpt, "split": split, "prior": "posthoc_magma" if posthoc else "own",
           "top1": float(np.mean(rank < 1)), "balanced_top1": float(recall[1].mean()),
           "top5": float(np.mean(rank < 5)), "balanced_top5": float(recall[5].mean()),
           "signs": [sign_names()[s] for s in present], "recall_top1": recall[1].tolist(),
           "recall_top5": recall[5].tolist()}
    print(f"{name} {split}: top-1 {100 * res['top1']:.1f} % (balanced {100 * res['balanced_top1']:.1f} %), "
          f"top-5 {100 * res['top5']:.1f} % (balanced {100 * res['balanced_top5']:.1f} %)", flush=True)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/{name}_{split}.json", "w") as f:
        json.dump(res, f, indent=1)


def table(out):
    """Table 1: balanced top-1 / top-5 (%) on testing."""
    rows = []
    for label, name in TABLE:
        path = f"{out}/{name}_testing.json"
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            rows.append(f"| {label} | {100 * r['balanced_top1']:.1f} | {100 * r['balanced_top5']:.1f} |")
        else:
            rows.append(f"| {label} | missing | missing |")
    print("Balanced accuracy (%) on LSFB100 testing, higher is better\n\n| Model | Top-1 | Top-5 |\n|---|---|---|")
    print("\n".join(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", nargs="?", help="checkpoint, without extension")
    parser.add_argument("--posthoc", action="store_true", help="post-hoc Magma prior on the frozen training codes")
    parser.add_argument("--split", default="testing", choices=("testing", "validation"))
    parser.add_argument("--out", default="results")
    parser.add_argument("--table", action="store_true", help="print Table 1 from the json files in --out")
    args = parser.parse_args()
    if args.table:
        table(args.out)
    elif args.ckpt:
        evaluate(args.ckpt, args.posthoc, args.split, args.out)
    else:
        parser.error("a checkpoint or --table is required")


if __name__ == "__main__":
    main()
