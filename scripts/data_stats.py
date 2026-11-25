"""Statistics of LSFB100 (data/lsfb100.npz), as given in the paper: clips, frames, signers, clip lengths, per split.

Usage: python scripts/data_stats.py
"""
import numpy as np

from mtgpvae.data import NPZ, SPLITS

f = np.load(NPZ)
print(f"{'':10s} {'clips':>7s} {'frames':>8s} {'signers':>8s} {'frames / clip':>16s} {'clips / sign':>14s} "
      f"{'mirrored':>9s}")
total = {"clips": 0, "frames": 0, "signers": set()}
for split in SPLITS:
    n, s = f[f"{split}_lengths"], f[f"{split}_signs"]
    per_sign = np.bincount(s)
    signers = set(f[f"{split}_signers"].tolist())
    print(f"{split:10s} {len(n):7d} {n.sum():8d} {len(signers):8d} {f'{n.min()} / {np.median(n):g} / {n.max()}':>16s} "
          f"{f'{per_sign.min()} - {per_sign.max()}':>14s} {f[f'{split}_mirror'].mean():9.1%}")
    total["clips"] += len(n)
    total["frames"] += int(n.sum())
    total["signers"] |= signers
print(f"{'all':10s} {total['clips']:7d} {total['frames']:8d} {len(total['signers']):8d}")
print("frames / clip: min / median / max. mirrored: left-dominant clips (flipped in training only)")
