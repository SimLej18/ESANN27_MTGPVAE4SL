"""Build the VAE pre-training frames (data/cont_frames.npy) from a local copy of LSFB-CONT (`just pull-cont`).

Frames of the unannotated LSFB-CONT recordings, 1 frame in CONT_STRIDE, normalised as LSFB100 (mtgpvae.data.normalise),
except the recordings that LSFB100 validation clips were cut from. Needs data/lsfb100.npz (`just data`) for that list.
Output: (n_frames, 274) float32, ~7.5 M frames (~8 GB).

Usage: python scripts/make_cont.py [--root ~/datasets/lsfb/lsfb-cont]
"""
import argparse
import os

import numpy as np
from sldl import SignLanguageDataset
from sldl.configs import LSFBContConfig

from mtgpvae.data import CONT, NPZ, normalise
from mtgpvae.pose import PARTS

SHARDS = range(9)  # unannotated shards
STRIDE = 3  # keep 1 frame in 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="~/datasets/lsfb/lsfb-cont", help="local copy of LSFB-CONT")
    root = os.path.expanduser(parser.parse_args().root)

    excluded = {"_".join(i.split("_")[:3]) for i in np.load(NPZ)["validation_ids"]}  # e.g. CLSFBI3804A_S078_B
    frames = []
    for shard in SHARDS:
        ds = SignLanguageDataset.from_config(LSFBContConfig(
            root=root, split="training", targets=None, show_loading_progress=False,
            shards_url=f"{root}/shards/unannotated/shard_{shard:06d}.tar"))
        for i in range(len(ds)):
            s = ds[i]
            if s["id"] not in excluded:
                poses = np.concatenate([s["poses"][p][..., :2] for p in PARTS], axis=1)
                frames.append(normalise(poses)[::STRIDE])
        print(f"shard {shard}: {sum(map(len, frames))} frames so far", flush=True)
    np.save(CONT, np.concatenate(frames))
    print(f"written {CONT}")


if __name__ == "__main__":
    main()
