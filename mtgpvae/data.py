"""LSFB100 clips, ready for the models (data/lsfb100.npz, built by scripts/make_lsfb100.py), and LSFB-CONT frames for
the VAE pre-training (data/cont_frames.npy, built by scripts/make_cont.py).

Poses: x, y only, centred on the shoulders and scaled to a unit mean shoulder width (per clip).
Time: frame i of a clip of n frames sits at t = i / (n - 1), mapped to the nearest of the G points of a regular grid on
[0, 1] (mimosa RegularGrid + NearestInputMapper). G = MAX_FRAMES, so two frames of a clip never share a grid point.
Clips are padded to N = MAX_FRAMES frames; grid index -1 marks a padding frame.
Mirror: left-dominant clips are flipped (dominant hand on the right) in the training split; in the validation and
testing splits, the flag is a label-free guess (left hand higher and travelling more) and is NOT applied by `load`.
"""
import os

import numpy as np
from sign_language_tools.pose.transforms import CenterOnLandmarks, NormalizeByReferenceEdge

from mtgpvae.pose import N_LANDMARKS, mirror

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV, SIGNS, NPZ, CONT = (os.path.join(ROOT, f) for f in ("lsfb100.csv", "signs.txt", "lsfb100.npz", "cont_frames.npy"))
SPLITS = ("training", "validation", "testing")
MIN_FRAMES, MAX_FRAMES = 8, 48
N = G = MAX_FRAMES  # frames per padded clip, grid points
SHOULDERS = (11, 12)  # MediaPipe pose landmarks (upper_pose comes first)

_center, _scale = CenterOnLandmarks(SHOULDERS), NormalizeByReferenceEdge(SHOULDERS)


def signs():
    """The 100 sign labels, in the order of the sign indices."""
    with open(SIGNS) as f:
        return f.read().split()


def normalise(poses):
    """Raw poses of one clip (T, 137, 2) -> flattened shoulder-centred frames (T, 274) float32."""
    return _scale(_center(poses.astype(np.float32))).reshape(len(poses), N_LANDMARKS * 2)


def grid_index(lengths):
    """Grid index (n, N) of every frame of clips of `lengths` frames, -1 on padding frames."""
    import jax.numpy as jnp
    from mimosa import NearestInputMapper, RegularGrid
    lengths = np.asarray(lengths)
    valid = np.arange(N) < lengths[:, None]
    t = np.where(valid, np.arange(N) / np.maximum(lengths[:, None] - 1, 1), np.nan)
    m = np.asarray(RegularGrid(jnp.asarray(t[..., None]), bounds=((0.0, 1.0),), n_points=G,
                               input_mapper=NearestInputMapper()).mappings)
    idx = np.where(valid, m, -1)
    assert ((idx >= 0) == valid).all() and (np.diff(np.where(valid, idx, G + np.arange(N)), axis=1) > 0).all()
    return idx


def load(split):
    """dict: x (n, N, 274) float32 (zero padding), lengths (n,), idx (n, N) grid index, signs, signers, ids, mirror
    (applied to x on the training split only, see the module docstring)."""
    if not os.path.exists(NPZ):
        raise FileNotFoundError(f"{NPZ} missing: run `just data` first")
    f = np.load(NPZ)
    lengths = f[f"{split}_lengths"]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    poses = f[f"{split}_poses"]
    x = np.zeros((len(lengths), N, N_LANDMARKS * 2), np.float32)
    for j, (a, n) in enumerate(zip(starts, lengths)):
        x[j, :n] = normalise(poses[a:a + n])
    m = f[f"{split}_mirror"]
    if split == "training":
        x = np.where(m[:, None, None], mirror(x), x)
    return {"x": x, "lengths": lengths, "idx": grid_index(lengths), "signs": f[f"{split}_signs"],
            "signers": f[f"{split}_signers"], "ids": f[f"{split}_ids"], "mirror": m}



def load_cont():
    """VAE pre-training frames (n, 274) float32, see scripts/make_cont.py."""
    if not os.path.exists(CONT):
        raise FileNotFoundError(f"{CONT} missing: run `just data-cont` first")
    return np.load(CONT)
