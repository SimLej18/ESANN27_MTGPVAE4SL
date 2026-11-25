"""Frame-level body-part VAE, pre-trained on LSFB-CONT (scripts/pretrain_vae.py) and fine-tuned with each prior.

One encoder / decoder per body part, each with its own latent dims (order of PARTS, default 12 / 8 / 8 / 4, d = 32):
- body: upper_pose (23 points) + 3 anchors: left hand wrist, right hand wrist, face centre. 52 values;
- left_hand, right_hand: the 20 other points of the hand, relative to the hand's own wrist (MediaPipe hand landmark 0),
  then rotated into the palm frame: the x axis is the body's palm direction (pose wrist -> middle of pose pinky and
  index). 40 values each;
- face: lips + eyes (72 points), relative to their centre. 144 values.
A frame is rebuilt from its parts (anchor + relative shape). At decoding, each hand is rotated back by the palm angle of
the DECODED body, so the hand's orientation follows the arm instead of being generated on its own.

Loss of one frame (negative ELBO, N(0, I) prior): squared errors of each part in its own space + BETA * KL, where the
hand targets are rotated by the decoded body's palm angle ("palm loss C"): the hand error is the error of the rebuilt
hand, and the body gets a gradient to reproduce the palm direction. Encoder inputs use the true palm angle.
"""
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mtgpvae.pose import SLICES

PARTS = ("body", "left_hand", "right_hand", "face")
SIZES = {"body": 52, "left_hand": 40, "right_hand": 40, "face": 144}
HANDS = {"left_hand": 1, "right_hand": 2}  # position in PARTS
DIMS = (12, 8, 8, 4)
WIDTHS = [256, 128, 64]  # encoder stages of every part; decoders use them in reverse order
SIGMA = 0.01  # assumed decoder noise, in shoulder widths
BETA = 2 * SIGMA**2
LOGVAR_RANGE = (-10.0, 2.0)
FACE = slice(SLICES["lips"].start, SLICES["right_eye"].stop)  # lips and eyes are contiguous
POSE_PALM = {"left_hand": (15, 17, 19), "right_hand": (16, 18, 20)}  # pose wrist, pinky, index (upper_pose indices)


# ---------------------------------------------------------------- frame <-> parts
def palm_rotation(up, h):
    """(cos, sin) of the palm direction of hand h, from the upper_pose points up (23, 2)."""
    w, pk, ix = (up[i] for i in POSE_PALM[h])
    v = (pk + ix) / 2 - w
    v = v / jnp.sqrt(jnp.sum(v**2) + 1e-8)
    return v[0], v[1]


def rotate(q, cs, inverse=False):
    """Rotates points q (n, 2) by -angle (into the palm frame), or by +angle (inverse)."""
    c, s = cs
    s = -s if inverse else s
    return jnp.stack([q[:, 0] * c + q[:, 1] * s, -q[:, 0] * s + q[:, 1] * c], axis=1)


def to_parts(x, palm_from=None):
    """Frame (274,) -> part vectors, order of PARTS. Hands are rotated by the palm angle of `palm_from` (upper_pose
    points (23, 2)), by default the frame's own body."""
    p = x.reshape(-1, 2)
    up = p[SLICES["upper_pose"]]
    palm_from = up if palm_from is None else palm_from
    lw, rw = p[SLICES["left_hand"].start], p[SLICES["right_hand"].start]
    face = p[FACE]
    fc = face.mean(0)
    body = jnp.concatenate([up, jnp.stack([lw, rw, fc])])

    def hand(h, w):
        return rotate(p[SLICES[h]][1:] - w, palm_rotation(palm_from, h)).ravel()
    return body.ravel(), hand("left_hand", lw), hand("right_hand", rw), (face - fc).ravel()


def from_parts(body, lh, rh, face):
    """Inverse of to_parts: hands rotated back by the palm angle of `body`."""
    b = body.reshape(-1, 2)
    up, lw, rw, fc = b[:-3], b[-3], b[-2], b[-1]

    def hand(q, w, h):
        return jnp.concatenate([w[None], rotate(q.reshape(-1, 2), palm_rotation(up, h), inverse=True) + w])
    return jnp.concatenate([up, hand(lh, lw, "left_hand"), hand(rh, rw, "right_hand"), face.reshape(-1, 2) + fc]).ravel()


# ---------------------------------------------------------------- networks
class StageResMLP(eqx.Module):
    """Per stage: Linear to the stage width + one pre-LayerNorm residual block. Then LayerNorm -> Linear out."""
    stages: list
    norm_out: eqx.nn.LayerNorm
    proj_out: eqx.nn.Linear

    def __init__(self, d_in, d_out, widths, key):
        keys = jax.random.split(key, 3 * len(widths) + 1)
        self.stages = []
        for i, (w_in, w) in enumerate(zip([d_in] + widths[:-1], widths)):
            k = keys[3 * i:3 * (i + 1)]
            self.stages.append((eqx.nn.Linear(w_in, w, key=k[0]), eqx.nn.LayerNorm(w), eqx.nn.Linear(w, w, key=k[1]),
                                eqx.nn.Linear(w, w, key=k[2])))
        self.norm_out = eqx.nn.LayerNorm(widths[-1])
        self.proj_out = eqx.nn.Linear(widths[-1], d_out, key=keys[-1])

    def __call__(self, x):
        h = x
        for proj, norm, lin1, lin2 in self.stages:
            h = proj(h)
            h = h + lin2(jax.nn.gelu(lin1(norm(h))))
        return self.proj_out(self.norm_out(h))


class PartCoder(eqx.Module):
    """Encoder / decoder of one part, inputs centred on the part's training mean."""
    encoder: StageResMLP
    decoder: StageResMLP
    mean: jax.Array

    def __init__(self, d_in, d_latent, mean, key):
        k_enc, k_dec = jax.random.split(key)
        self.encoder = StageResMLP(d_in, 2 * d_latent, WIDTHS, k_enc)
        self.decoder = StageResMLP(d_latent, d_in, WIDTHS[::-1], k_dec)
        self.mean = mean

    def encode(self, x):
        mu, logvar = jnp.split(self.encoder(x - jax.lax.stop_gradient(self.mean)), 2)
        return mu, jnp.clip(logvar, *LOGVAR_RANGE)

    def decode(self, z):
        return self.decoder(z) + jax.lax.stop_gradient(self.mean)


class PartVAE(eqx.Module):
    """Acts on a single frame (274,): use jax.vmap for batches. Codes are concatenated in the order of PARTS."""
    coders: tuple
    dims: tuple = eqx.field(static=True)

    def __init__(self, dims, means, key):
        """means: tuple of part means (training frames through to_parts)."""
        self.dims = tuple(int(d) for d in dims)
        keys = jax.random.split(key, len(PARTS))
        self.coders = tuple(PartCoder(SIZES[p], d, m, k) for p, d, m, k in zip(PARTS, self.dims, means, keys))

    def split(self, z):
        return jnp.split(z, np.cumsum(self.dims)[:-1])

    def encode_parts(self, parts):
        mus, logvars = zip(*[c.encode(x) for c, x in zip(self.coders, parts)])
        return jnp.concatenate(mus), jnp.concatenate(logvars)

    def decode_parts(self, z):
        return tuple(c.decode(zp) for c, zp in zip(self.coders, self.split(z)))

    def encode(self, x):
        return self.encode_parts(to_parts(x))

    def decode(self, z):
        return from_parts(*self.decode_parts(z))


def part_means(frames, batch=100_000):
    """Mean of each part over training frames (N, 274)."""
    f = jax.jit(jax.vmap(to_parts))
    sums = [0.0] * len(PARTS)
    for i in range(0, len(frames), batch):
        parts = f(jnp.asarray(frames[i:i + batch], jnp.float32))
        sums = [s + np.asarray(p, np.float64).sum(0) for s, p in zip(sums, parts)]
    return tuple(jnp.asarray(s / len(frames), jnp.float32) for s in sums)


def loss(model, x, key):
    """Negative ELBO of one frame (up to constants), each part's error in its own space, hand targets in the palm
    frame of the decoded body (palm loss C): (loss, (reconstruction, kl))."""
    mu, logvar = model.encode(x)
    z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
    dec = model.decode_parts(z)
    target = to_parts(x, palm_from=dec[0].reshape(-1, 2)[:-3])
    rec = sum(jnp.sum((r - t) ** 2) for r, t in zip(dec, target))
    kl = 0.5 * jnp.sum(mu**2 + jnp.exp(logvar) - 1 - logvar)
    return rec + BETA * kl, (rec, kl)


# ---------------------------------------------------------------- checkpoints
def save(path, model, info):
    """Writes path.eqx and path.json (info must contain "dims")."""
    eqx.tree_serialise_leaves(f"{path}.eqx", model)
    with open(f"{path}.json", "w") as f:
        json.dump(info, f)


def load(path):
    """Inverse of save: (model, info), weights in float32 (also when x64 is on)."""
    with open(f"{path}.json") as f:
        info = json.load(f)
    like = PartVAE(info["dims"], tuple(np.zeros(SIZES[p], np.float32) for p in PARTS), jax.random.key(0))
    like = jax.tree.map(lambda a: a.astype(jnp.float32) if eqx.is_inexact_array(a) else a, like)
    return eqx.tree_deserialise_leaves(f"{path}.eqx", like), info
