"""Inference with a trained model, on whole splits: codes, sign posteriors, classification, post-hoc Magma prior.

Codes: zhat, s2 = encoder outputs at the real frames, on the G-point grid (y = zhat - m0, lam = 1 / s2, 0 without a
frame). Sign statistics: each sign's q(mu_s) from all its training clips, under the prior as trained.
Classification: log N(zhat; m_k, C_k + k_clip + D) on the clip's grid points, D = the clip's own s2, q(mu_k) = (m_k, C_k).
Post-hoc Magma: a (traj, gp) prior fitted by maximum likelihood (LBFGS) on the frozen training codes of any model.
Generation: mu_s ~ q(mu_s), e ~ GP(0, k_clip without its white noise), decoded.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from mimosa.nll import single_channel_mvn_nll

from mtgpvae.data import G, MIN_FRAMES, N
from mtgpvae.model import EYE, PATTERNS, _clip, _mean, chols, onehot, to_grid
from mtgpvae.prior import F64, Prior

CHUNK, Q_CHUNK = 256, 8  # clips per call: encoding / sign statistics, classification
BLOCK = 32  # clips per block in the post-hoc fit


# ---------------------------------------------------------------- codes
def encode(vae, x):
    """Encoder means and log-variances (n, N, d) of clips x (n, N, 274)."""
    f = eqx.filter_jit(lambda v, a: jax.vmap(jax.vmap(v.encode))(a))
    out = [f(vae, jnp.asarray(x[i:i + CHUNK])) for i in range(0, len(x), CHUNK)]
    return tuple(np.concatenate([np.asarray(o[k]) for o in out]) for k in (0, 1))


def decode(vae, z):
    """Decoded frames (n, N, 274) of codes z (n, N, d)."""
    f = eqx.filter_jit(lambda v, a: jax.vmap(jax.vmap(v.decode))(a.astype(jnp.float32)))
    return np.concatenate([np.asarray(f(vae, jnp.asarray(z[i:i + CHUNK]))) for i in range(0, len(z), CHUNK)])


def grid_codes(m0, mu, logvar, idx):
    """y = zhat - m0 and lam = 1 / s2 on the grid (n, d, G) (0 without a frame)."""
    y, lam = to_grid(m0, jnp.asarray(mu)[:, None], jnp.asarray(logvar)[:, None], onehot(jnp.asarray(idx))[:, None])
    return np.asarray(y[:, :, 0]), np.asarray(lam[:, :, 0])


# ---------------------------------------------------------------- sign posteriors
@eqx.filter_jit
def _stats(ls, y, lam):
    b = jax.vmap(jax.vmap(lambda l, lm: _clip(l, lm)[2]), (None, 0))(ls, lam)  # (c, d, G, G)
    return b, jnp.einsum("ndij,ndj->ndi", b, y)


def sign_stats(ls, y, lam, signs, n_signs):
    """Summed B^-1 (S, d, G, G) and B^-1 y (S, d, G) of the clips of every sign, ls = chol(k_clip)."""
    g, h = np.zeros((n_signs, y.shape[1], G, G)), np.zeros((n_signs, y.shape[1], G))
    for i in range(0, len(y), CHUNK):
        b, by = (np.asarray(a) for a in _stats(ls, jnp.asarray(y[i:i + CHUNK]), jnp.asarray(lam[i:i + CHUNK])))
        np.add.at(g, signs[i:i + CHUNK], b)
        np.add.at(h, signs[i:i + CHUNK], by)
    return g, h


@eqx.filter_jit
def sign_means(l0, g, h):
    """q(mu_s) from the sign statistics: means (S, d, G), covs (S, d, G, G)."""
    _, vbar, a = jax.vmap(jax.vmap(_mean, (0, 0, 0)), (None, 0, 0))(l0, g, h)
    return jnp.einsum("dij,sdj->sdi", l0, vbar), jnp.einsum("sdki,sdkj->sdij", a, a)


# ---------------------------------------------------------------- generation
def generate(prior, vae, stats, lengths, n_gen, key):
    """n_gen clips of every sign, of `lengths` (S,) frames, from the sign statistics stats = (g, h):
    (S n_gen, N, 274) frames, sign by sign, and their lengths (S n_gen,)."""
    l0 = np.asarray(chols(prior)[0])
    k_clip = np.asarray(prior.covs()[1])  # without the white noise
    m, cov = (np.asarray(a) for a in sign_means(jnp.asarray(l0), jnp.asarray(stats[0]), jnp.asarray(stats[1])))
    k1, k2 = jax.random.split(key)
    d, n_signs = m.shape[1], m.shape[0]
    chol = lambda a: np.linalg.cholesky(a + 1e-8 * np.eye(G))
    mu = m[:, None] + np.einsum("sdij,sgdj->sgdi", chol(cov), np.asarray(jax.random.normal(k1, (n_signs, n_gen, d, G))))
    e = np.einsum("dij,sgdj->sgdi", chol(k_clip), np.asarray(jax.random.normal(k2, (n_signs, n_gen, d, G))))
    z = (mu + e).reshape(-1, d, G)
    lg = np.repeat(lengths, n_gen)
    idx = np.asarray(PATTERNS[lg - MIN_FRAMES].argmax(-1))
    idx = np.where(np.arange(N) < lg[:, None], idx, -1)
    z_frames = np.einsum("nkg,ndg->nkd", np.asarray(onehot(jnp.asarray(idx))), z)  # at the real frames
    return decode(vae, z_frames + np.asarray(prior.m())), lg


# ---------------------------------------------------------------- classification
@eqx.filter_jit
def _nll(y, s2, m, cov, k_clip):
    """NLL (Q, S) of the queries y (NaN without a frame), s2 (Q, d, G) under every sign (m, cov (S, d, ...))."""
    def one(yq, s2q):
        dm = s2q[:, :, None] * EYE
        return jax.vmap(lambda mk, ck: jnp.sum(jax.vmap(single_channel_mvn_nll)(yq, mk, ck + k_clip + dm)))(m, cov)
    return jax.vmap(one)(y, s2)


def class_ll(k_clip, means, y, lam):
    """Log-likelihood (Q, S) of Q clips' codes y, lam (Q, d, G) under every sign, means = sign_means(...)."""
    m, cov = (jnp.asarray(a) for a in means)
    yn = np.where(lam > 0, y, np.nan)
    s2 = np.where(lam > 0, 1 / np.where(lam > 0, lam, 1), 1.0)
    return -np.concatenate([np.asarray(_nll(jnp.asarray(yn[i:i + Q_CHUNK]), jnp.asarray(s2[i:i + Q_CHUNK]), m, cov,
                                            jnp.asarray(k_clip))) for i in range(0, len(y), Q_CHUNK)])


# ---------------------------------------------------------------- post-hoc Magma: maximum likelihood on frozen codes
# Y ~ N(0, 11^T (x) k_sign + blockdiag(k_clip + D_r)) on the observed points of the n clips of one sign, one dim, splits
# into per-clip terms (summed over the sign's clips) and one term of the sign's mean process, computed from those sums.
# Clips go in blocks of BLOCK clips of the same sign.
def _clip_terms(ls, y, lam):
    """Clips y, lam (n, G) of one sign, one dim (lam = 0: no frame): summed B^-1 (G, G), summed B^-1 y (G,), and
    sum of y^T B^-1 y + log det(k_clip + D_r) on the observed points (log det via chol(I + ls^T diag(lam) ls))."""
    ln, _, binv = jax.vmap(_clip, (None, 0))(ls, lam)
    by = jnp.einsum("nij,nj->ni", binv, y)
    obs = lam > 0
    c = (jnp.sum(y * by) + 2 * jnp.sum(jnp.log(jnp.diagonal(ln, axis1=-2, axis2=-1)))
         - jnp.sum(jnp.where(obs, jnp.log(jnp.where(obs, lam, 1.0)), 0.0)))
    return binv.sum(0), by.sum(0), c


def _sign_term(l0, g, h):
    """Mean-process term of one sign, one dim, from its clips' summed g = sum B^-1, h = sum B^-1 y."""
    lm, vbar, _ = _mean(l0, g, h)
    return -(l0.T @ h) @ vbar + 2 * jnp.sum(jnp.log(jnp.diagonal(lm)))


def magma_nll(prior, zs, lams, block_sign, n_signs):
    """NLL per observed point of every sign's codes: zs = zhat, lams (B, d, BLOCK, G) in blocks of clips of one sign
    (the last block of each sign completed with lam = 0), block_sign (B,) the sign of each block."""
    l0, ls, _ = chols(prior)
    m0 = prior.m()

    @jax.checkpoint
    def one(args):
        z, lam = args
        y = jnp.where(lam > 0, z - m0[:, None, None], 0.0)
        return jax.vmap(_clip_terms)(ls, y, lam)
    g, h, c = jax.lax.map(one, (zs, lams))  # (B, d, G, G), (B, d, G), (B, d)
    g, h = (jax.ops.segment_sum(a, block_sign, n_signs) for a in (g, h))
    sign = jax.vmap(jax.vmap(_sign_term), (None, 0, 0))(l0, g, h)  # (S, d)
    n_obs = jnp.sum(lams > 0)
    return 0.5 * (jnp.sum(c) + jnp.sum(sign) + n_obs * jnp.log(2 * jnp.pi)) / n_obs


def sign_blocks(z, lam, signs, n_signs):
    """Codes z, lam (n, d, G) -> blocks (B, d, BLOCK, G) of clips of one sign, and the sign of each block (B,)."""
    zb, lb, sb = [], [], []
    for s in range(n_signs):
        j = np.flatnonzero(signs == s)
        for a in range(0, len(j), BLOCK):
            k = j[a:a + BLOCK]
            zz, ll = np.zeros((z.shape[1], BLOCK, G)), np.zeros((z.shape[1], BLOCK, G))
            zz[:, :len(k)], ll[:, :len(k)] = np.moveaxis(z[k], 0, 1), np.moveaxis(lam[k], 0, 1)
            zb.append(zz), lb.append(ll), sb.append(s)
    return np.stack(zb), np.stack(lb), np.array(sb)


def fit_magma(z, lam, signs, n_signs, max_steps=500):
    """Magma prior (traj, gp) by maximum likelihood (LBFGS, every HP and m0) on frozen codes zhat, lam (n, d, G) on the
    grid; starts from Prior.init with m0 = the mean code. Returns (prior, NLL per observed point, LBFGS steps taken)."""
    m0 = np.sum(z * (lam > 0), (0, 2)) / np.sum(lam > 0, (0, 2))
    prior = Prior.init(z.shape[1], "traj", "gp")
    prior = eqx.tree_at(lambda p: p.m0.inner.inner.constant, prior, jnp.asarray(m0, F64))
    params, static = eqx.partition(prior, eqx.is_inexact_array)
    zs, lams, block_sign = sign_blocks(z, lam, signs, n_signs)
    data = (jnp.asarray(zs), jnp.asarray(lams), jnp.asarray(block_sign))
    obj = eqx.filter_jit(lambda p, a: magma_nll(eqx.combine(p, static), *a, n_signs))
    sol = optx.minimise(obj, optx.LBFGS(rtol=1e-6, atol=1e-6), params, data, max_steps=max_steps, throw=False)
    return eqx.combine(sol.value, static), float(obj(sol.value, data)), int(sol.stats["num_steps"])
