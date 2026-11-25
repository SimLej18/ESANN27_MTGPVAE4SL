"""Build LSFB100 (data/lsfb100.npz) from a local copy of LSFB-ISOL (`just pull-isol`).

LSFB100: the clips listed in data/lsfb100.csv, i.e. the 100 most frequent signs of LSFB-ISOL, clips of 8 to 48 frames.
Splits are LSFB-ISOL's (testing: shard 0, validation: shards 1-2, training: shards 3-9); on the training split, clips
too atypical for their sign and two signers were also removed. The `mirror` column marks left-dominant clips.

Poses are kept raw (x, y of the 137 landmarks, LSFB-ISOL coordinates); mtgpvae.data normalises them when loading.
For each split, data/lsfb100.npz holds {split}_poses (frames of all clips, concatenated, (n_frames, 137, 2), float16
as in LSFB-ISOL), {split}_lengths, {split}_ids, {split}_signs (index in data/signs.txt), {split}_signers, {split}_mirror.

Usage: python scripts/make_lsfb100.py [--root ~/datasets/lsfb/lsfb-isol]
"""
import argparse
import csv
import os

import numpy as np
from sldl import SignLanguageDataset
from sldl.configs import LSFBIsolConfig

from mtgpvae.data import CSV, NPZ, SIGNS, SPLITS
from mtgpvae.pose import PARTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="~/datasets/lsfb/lsfb-isol", help="local copy of LSFB-ISOL")
    root = os.path.expanduser(parser.parse_args().root)

    with open(SIGNS) as f:
        index = {s: i for i, s in enumerate(f.read().split())}
    with open(CSV) as f:
        rows = list(csv.DictReader(f))

    out = {}
    for split in SPLITS:
        clips = [r for r in rows if r["split"] == split]
        pos = {r["id"]: j for j, r in enumerate(clips)}
        found = [None] * len(clips)
        ds = SignLanguageDataset.from_config(LSFBIsolConfig(root=root, variant="500", split=split,
                                                                   show_loading_progress=False))
        for i in range(len(ds)):
            s = ds[i]
            j = pos.get(s["id"])
            if j is not None:
                assert s["label"] == clips[j]["sign"], f"{s['id']}: label {s['label']}, expected {clips[j]['sign']}"
                poses = np.concatenate([s["poses"][p][..., :2] for p in PARTS], axis=1)
                found[j] = (poses, s["signer_id"])
        missing = [c["id"] for c, f in zip(clips, found) if f is None]
        assert not missing, f"{len(missing)} {split} clips not found in {root}, e.g. {missing[:3]}"
        out[f"{split}_poses"] = np.concatenate([f[0] for f in found])
        out[f"{split}_lengths"] = np.array([len(f[0]) for f in found])
        out[f"{split}_ids"] = np.array([c["id"] for c in clips])
        out[f"{split}_signs"] = np.array([index[c["sign"]] for c in clips])
        out[f"{split}_signers"] = np.array([f[1] for f in found])
        out[f"{split}_mirror"] = np.array([c["mirror"] == "1" for c in clips])
        print(f"{split}: {len(clips)} clips, {len(out[f'{split}_poses'])} frames")
    np.savez(NPZ, **out)
    print(f"written {NPZ}")


if __name__ == "__main__":
    main()
