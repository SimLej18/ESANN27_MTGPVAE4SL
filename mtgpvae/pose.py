"""Skeleton layout: 137 MediaPipe landmarks (LSFB body parts, concatenated), x/y only, flattened to 274 values."""
import numpy as np
from sign_language_tools.pose.mediapipe.edges import EYE_EDGES, HAND_EDGES, LIPS_EDGES, UPPER_POSE_EDGES

SIZES = {"upper_pose": 23, "left_hand": 21, "right_hand": 21, "lips": 40, "left_eye": 16, "right_eye": 16}
PARTS = list(SIZES)
N_LANDMARKS = sum(SIZES.values())  # 137
_starts = np.cumsum([0] + list(SIZES.values()))
SLICES = {p: slice(_starts[i], _starts[i + 1]) for i, p in enumerate(PARTS)}  # landmark indices of each part
EDGES = {"upper_pose": UPPER_POSE_EDGES, "left_hand": HAND_EDGES, "right_hand": HAND_EDGES,
         "lips": LIPS_EDGES, "left_eye": EYE_EDGES, "right_eye": EYE_EDGES}  # skeleton edges, indices inside each part

# Left-right mirror. MediaPipe pose: nose 0, then left / right pairs. Lips: 0-3 on the midline, 4-21 <-> 22-39.
# Eyes and hands: same index in the other part.
POSE_MIRROR = [0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21]
LIPS_MIRROR = [0, 1, 2, 3] + list(range(22, 40)) + list(range(4, 22))
_PAIRS = {"upper_pose": ("upper_pose", POSE_MIRROR), "lips": ("lips", LIPS_MIRROR),
          "left_hand": ("right_hand", None), "right_hand": ("left_hand", None),
          "left_eye": ("right_eye", None), "right_eye": ("left_eye", None)}
MIRROR = np.concatenate([np.arange(SLICES[o].start, SLICES[o].stop)[perm if perm is not None else slice(None)]
                         for o, perm in (_PAIRS[p] for p in PARTS)])  # landmark i of the mirror = landmark MIRROR[i]


def mirror(x):
    """Left-right mirror of shoulder-centred flattened frames x (..., 274): x -> -x, left / right landmarks swapped."""
    p = x.reshape(*x.shape[:-1], N_LANDMARKS, 2)[..., MIRROR, :] * np.array([-1, 1], x.dtype)
    return p.reshape(x.shape)
