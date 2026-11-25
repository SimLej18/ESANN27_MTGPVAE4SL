"""Latent prior of the three models, per latent dim, on the G-point grid of mtgpvae.data.

Clip r of sign s, time t in [0, 1] (clip durations), one prior per latent dim (HPs per dim, shared by every sign):
    z_r(t) = m0 + mu_s(t) + e_r(t),   mu_s ~ GP(0, k_sign),   e_r ~ GP(0, k_clip)
- k_sign (K_s in the paper), what the sign imposes: "const" k_sign = c (a posture: mu_s is constant in time);
  "traj" k_sign = c + sm2 M52(lm) (a trajectory);
- k_clip (K_c in the paper), how a clip varies around it: "white" k_clip = sw2 I; "gp" k_clip = off + sd2 M52(ld) + sw2 I.
  The learned white noise sw2 I is left out of generated clips.
The three models are special cases of this one prior; the first two are the usual label-conditioned VAEs:
- sign-VAE (const, white): z_r(t) = m0 + mu_s + noise, frames independent given the sign. A VAE whose prior is a
  Gaussian mixture with one component per sign (a supervised / class-conditional VAE), the component's mean mu_s
  being inferred from the sign's clips instead of learned as a parameter;
- sign-GP-VAE (const, gp): z_r(t) = m0 + mu_s + e_r(t). A GP-VAE (a GP prior along time, one per latent dim) whose
  prior mean depends on the sign: the same per-sign constant, plus a smooth clip-specific trajectory around it;
- MTGP-VAE (traj, gp): the sign's mean is itself a trajectory mu_s(t), shared by all its clips (Magma's mean process).
Every component's variance is rescaled so that the total variance per dim is 1 - NOISE_MIN, then NOISE_MIN I is added.
The sign's mean process is never a parameter: it is inferred from the sign's clips (mtgpvae.model).

Kernels and means are kernax modules, batched over the latent dims with mimosa's builders (channel = latent dim).
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import kernax
import numpy as np
from mimosa import Dimensions, ModelConfig, RegularGrid
from mimosa.synthetic import build_mean, build_mean_kernel

from mtgpvae.data import G

NOISE_MIN = 1e-3
MODELS = {"sign_vae": ("const", "white"), "sign_gp_vae": ("const", "gp"), "mtgp_vae": ("traj", "gp")}
F64 = jnp.float64
V, M52 = kernax.VarianceKernel, kernax.Matern52Kernel
COMPONENTS = ("posture", "motion", "offset", "deviation", "white")  # k_sign: posture, motion; k_clip: offset, deviation, white


def safe_euclidean(x1, x2):
    """Euclidean distance with a finite gradient at distance 0 (kernax's default gives NaN there, Matern kernels)."""
    sq = jnp.sum((x1 - x2) ** 2)
    nz = sq > 0
    return jnp.where(nz, jnp.sqrt(jnp.where(nz, sq, 1.0)), 0.0)


def grid_points():
    """The G grid points (G, 1), float64."""
    return jnp.asarray(RegularGrid(bounds=((0.0, 1.0),), n_points=G).points, F64)


def _batched(kernel, d):
    """One copy of `kernel` per latent dim (HPs (d,)): a kernax module returning (1, d, G, G)."""
    dims = Dimensions(T=1, K=1, I=1, C=d, O=1, N=G, G=G)
    cfg = ModelConfig(shared_cluster_hps=True, cluster_specific_task_hps=False, shared_channel_hps=False)
    return build_mean(kernel, dims, cfg) if isinstance(kernel, kernax.ConstantMean) else build_mean_kernel(kernel, dims, cfg)


class Prior(eqx.Module):
    """Components are None when the model has none (e.g. motion for "const"). HPs in float64."""
    m0: kernax.AbstractModule
    posture: kernax.AbstractModule
    motion: kernax.AbstractModule | None
    offset: kernax.AbstractModule | None
    deviation: kernax.AbstractModule | None
    white: kernax.AbstractModule
    sign_kind: str = eqx.field(static=True)
    clip_kind: str = eqx.field(static=True)

    @classmethod
    def init(cls, d, sign_kind, clip_kind, posture=0.15, motion=0.15, motion_ls=0.2, offset=0.3, deviation=0.3,
             deviation_ls=0.2, white=0.05):
        """HP values: scalars, the same for every dim (mimosa's builders copy them per dim). Variances are relative
        (rescaled to a total of 1)."""
        assert (sign_kind, clip_kind) in MODELS.values()
        f = float
        m52 = lambda ls: M52(length_scale=f(ls), distance_function=safe_euclidean)
        gp = clip_kind == "gp"
        return cls(m0=_batched(kernax.ConstantMean(f(0.0)), d),
                   posture=_batched(V(f(posture)), d),
                   motion=_batched(V(f(motion)) * m52(motion_ls), d) if sign_kind == "traj" else None,
                   offset=_batched(V(f(offset)), d) if gp else None,
                   deviation=_batched(V(f(deviation)) * m52(deviation_ls), d) if gp else None,
                   white=_batched(kernax.WhiteNoiseKernel(f(white)), d),
                   sign_kind=sign_kind, clip_kind=clip_kind)

    def _parts(self, u):
        """Unscaled covariance of each present component on points u: {name: (d, n, n)}."""
        return {c: getattr(self, c)(u, u)[0] for c in COMPONENTS if getattr(self, c) is not None}

    def scale(self):
        """Per dim factor a (d,): a * (sum of the components' variances) = 1 - NOISE_MIN (stationary kernels)."""
        u = grid_points()[:1]
        return (1 - NOISE_MIN) / sum(k[:, 0, 0] for k in self._parts(u).values())

    def m(self):
        """Constant mean m0 (d,)."""
        return self.m0(grid_points()[:1])[0, :, 0]

    def covs(self):
        """k_sign, k_clip without white noise, k_clip used in the posterior (white + NOISE_MIN included): (d, G, G) each."""
        u = grid_points()
        p, a = self._parts(u), self.scale()[:, None, None]
        zero = jnp.zeros((a.shape[0], G, G), F64)
        k_sign = a * sum((p[c] for c in ("posture", "motion") if c in p), zero)
        k_clip = a * sum((p[c] for c in ("offset", "deviation") if c in p), zero)
        return k_sign, k_clip, k_clip + a * p["white"] + NOISE_MIN * jnp.eye(G, dtype=F64)


def frozen(tree):
    """Same values, no gradient."""
    params, static = eqx.partition(tree, eqx.is_inexact_array)
    return eqx.combine(jax.lax.stop_gradient(params), static)


def to_json(prior):
    return [np.asarray(a).tolist() for a in jax.tree.leaves(eqx.filter(prior, eqx.is_inexact_array))]


def from_json(leaves, d, sign_kind, clip_kind):
    like = Prior.init(d, sign_kind, clip_kind)
    params, static = eqx.partition(like, eqx.is_inexact_array)
    return eqx.combine(jax.tree.unflatten(jax.tree.structure(params), [jnp.asarray(a, F64) for a in leaves]), static)
