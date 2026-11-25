"""Posterior and loss terms of the three models (VAE + MTGP prior of mtgpvae.prior).

Encoder = pseudo-observations: per real frame and latent dim, zhat with a known noise variance s2. q = exact posterior
of the prior given the pseudo-observations, on the G-point grid (a grid point without a frame adds nothing): the n clips
of a sign of the batch together give q(mu_s), then q(e_r | mu_s). The decoder gets one sample z ~ q.
Loss per real frame of the batch (a long clip weighs more than a short one):
    rec + beta * (KL_mean + KL_res) / frames + beta * ALPHA * CE / (frames per clip) + INTERP_WEIGHT * interp
- rec: squared error in pose space (274 values; the decoder rebuilds the hands with the decoded body's palm angle);
- KL_mean + KL_res: KL(q || prior) in closed form, mean process and clip residuals;
- CE: generative classifier among the batch signs, log N(zhat_r; m_k, C_k + k_clip + Dbar I) restricted to the clip's grid
  points (own sign: leave-one-out), Dbar = batch mean of s2 per dim, prior HPs frozen in this term. Optionally tempered
  (logits / ce_temp, weight times ce_temp);
- interp (models with a GP k_clip): each clip observed at its first two and last two frames only (plus the other clips of its
  sign), one posterior sample decoded, squared error on the hidden frames. Its gradient on the HPs is scaled by
  INTERP_HP.
Linear algebra in whitened form: l0 = chol(k_sign + JITTER), ls = chol(k_clip), never k_sign^-1.
Notation: k_sign and k_clip are the paper's K_s and K_c (mtgpvae.prior).
"""
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from mtgpvae.data import G, MAX_FRAMES, MIN_FRAMES, N, grid_index
from mtgpvae.prior import F64, frozen, from_json, to_json
from mtgpvae.vae import BETA, PARTS, SIZES, PartVAE

ALPHA = 1000.0  # CE weight, in units of beta per frame
INTERP_WEIGHT = 0.2
INTERP_HP = 0.0077  # gradient scale of interp on the HPs
JITTER = 5e-5  # on k_sign before its Cholesky
N_SIGNS, N_CLIPS = 32, 16  # batch: signs x clips per sign
EYE = jnp.eye(G, dtype=F64)
LENGTHS = np.arange(MIN_FRAMES, MAX_FRAMES + 1)
PATTERNS = jnp.asarray((grid_index(LENGTHS)[..., None] == np.arange(G)).astype(np.float64))  # (P, N, G) one-hot
TERMS = ["rec", "kl_mean", "kl_res", "ce", "interp"]


class Model(eqx.Module):
    vae: eqx.Module  # vae.PartVAE, float32
    prior: eqx.Module  # prior.Prior, float64

    @property
    def has_interp(self):
        return self.prior.clip_kind == "gp"


def chols(prior):
    """Per dim Choleskys of k_sign + JITTER and of k_clip, and k_clip: (d, G, G) each."""
    k_sign, _, k_clip = prior.covs()
    return jnp.linalg.cholesky(k_sign + JITTER * EYE), jnp.linalg.cholesky(k_clip), k_clip


def gscale(tree, w):
    """Same values, gradient multiplied by w."""
    params, static = eqx.partition(tree, eqx.is_inexact_array)
    params = jax.tree.map(lambda a: w * a + jax.lax.stop_gradient((1 - w) * a), params)
    return eqx.combine(params, static)


# ---------------------------------------------------------------- one sign, one dim
def _clip(ls, lam):
    """Clip given its mean process, whitened by ls = chol(k_clip), lam = 1 / s2 (0 on grid points without a frame):
    chol(I + ls^T diag(lam) ls), P, and B^-1 = (k_clip + diag(1 / lam))^-1."""
    ln = jnp.linalg.cholesky(EYE + ls.T @ (lam[:, None] * ls))
    p = jsl.cho_solve((ln, True), ls.T * lam)
    binv = jsl.solve_triangular(ls.T, p, lower=False)
    return ln, p, 0.5 * (binv + binv.T)


def _mean(l0, g, h):
    """q(mu) from the summed precisions g and h = sum B^-1 y: chol(M), vbar, a with mean l0 vbar, cov(mu) = a^T a."""
    lm = jnp.linalg.cholesky(EYE + l0.T @ g @ l0)
    return lm, jsl.cho_solve((lm, True), l0.T @ h), jsl.solve_triangular(lm, l0.T, lower=True)


def _kl_white(lc, sq):
    """KL(N(a, (lc lc^T)^-1) || N(0, I)), sq = E||a||^2."""
    inv = jsl.solve_triangular(lc, EYE, lower=True)
    return 0.5 * (jnp.sum(inv ** 2) + sq - G + 2 * jnp.sum(jnp.log(jnp.diagonal(lc))))


def _sign(l0, ls, y, lam, key):
    """n clips of one sign, one dim, y = zhat - m0 and lam (n, G): a sample z - m0, the KLs, q(mu) and clip stats."""
    ln, p, binv = jax.vmap(_clip, (None, 0))(ls, lam)
    by = jnp.einsum("nij,nj->ni", binv, y)
    lm, vbar, a = _mean(l0, binv.sum(0), by.sum(0))
    m = l0 @ vbar
    pr = jnp.einsum("nij,nj->ni", p, y - m)
    kl_mean = _kl_white(lm, vbar @ vbar)
    kl_res = jax.vmap(_kl_white)(ln, jnp.sum(pr ** 2, -1) + jnp.sum(jnp.einsum("nij,kj->nik", p, a) ** 2, (-2, -1)))
    k1, k2 = jax.random.split(key)
    mu = m + a.T @ jax.random.normal(k1, (G,), F64)
    noise = jax.vmap(lambda l, xi: jsl.solve_triangular(l.T, xi, lower=False))(ln, jax.random.normal(k2, y.shape, F64))
    e = (jnp.einsum("nij,nj->ni", p, y - mu) + noise) @ ls.T
    return {"z": mu + e, "kl_mean": kl_mean, "kl_res": kl_res, "m": m, "a": a, "binv": binv, "by": by}


def _loo(l0, binv, by):
    """Per clip, q(mu) from the other clips of its sign: means (n, G), covs (n, G, G)."""
    g, h = binv.sum(0), by.sum(0)

    def one(b, hb):
        _, vbar, a = _mean(l0, g - b, h - hb)
        return l0 @ vbar, a.T @ a
    return jax.vmap(one)(binv, by)


def _interp(l0, ls, y, lam, binv, by, obs, key):
    """One posterior sample (n, G) of z - m0 of each clip seen at its grid points obs (n, G) only, the other clips of
    its sign fully seen: mean process sample, then the clip's residual sample given it."""
    g, h = binv.sum(0), by.sum(0)

    def one(yi, li, b, hb, oi, k):
        ln, p, bm = _clip(ls, li * oi)
        _, vbar, a = _mean(l0, g - b + bm, h - hb + bm @ yi)
        k1, k2 = jax.random.split(k)
        mu = l0 @ vbar + a.T @ jax.random.normal(k1, (G,), F64)
        return mu + ls @ (p @ (yi - mu) + jsl.solve_triangular(ln.T, jax.random.normal(k2, (G,), F64), lower=False))
    return jax.vmap(one)(y, lam, binv, by, obs, jax.random.split(key, len(y)))


# ---------------------------------------------------------------- batch
def onehot(idx):
    """Grid indices (..., N) (-1: padding) -> one-hot (..., N, G) float64, zero rows on padding frames."""
    return (idx[..., None] == jnp.arange(G)).astype(F64)


def to_grid(m0, mu, logvar, oh):
    """Encoder outputs at the real frames (B, n, N, d) -> y = zhat - m0 and lam = 1 / s2 on the grid (B, d, n, G),
    both 0 on grid points without a frame."""
    y = jnp.einsum("bnkg,bnkd->bdng", oh, mu.astype(F64) - m0)
    lam = jnp.einsum("bnkg,bnkd->bdng", oh, jnp.exp(-logvar.astype(F64)))
    return y, lam


def to_frames(z, oh):
    """Codes on the grid (B, d, n, G) -> at the real frames (B, n, N, d) (0 on padding frames)."""
    return jnp.einsum("bnkg,bdng->bnkd", oh, z)


def clip_stats(ls, y, lam):
    """B^-1 (B, d, n, G, G) and B^-1 y (B, d, n, G) of every clip, ls = chol(k_clip) (d, G, G)."""
    binv = jax.vmap(jax.vmap(lambda l, lm: jax.vmap(_clip, (None, 0))(l, lm)[2], (0, 0)), (None, 0))(ls, lam)
    return binv, jnp.einsum("bdnij,bdnj->bdni", binv, y)


def batch_posterior(c, y, lam, key):
    """_sign over signs (B) and dims (d): y, lam (B, d, n, G), c = chols(prior) ((d, G, G) each)."""
    l0, ls, _ = c
    keys = jax.random.split(key, y.shape[:2])
    return jax.vmap(jax.vmap(_sign), (None, None, 0, 0, 0))(l0, ls, y, lam, keys)


def sq_err(vae, x, z):
    """Squared error per frame (..., N) of the decoded codes z (..., N, d) against x (..., N, 274)."""
    x_hat = jax.vmap(jax.vmap(jax.vmap(vae.decode)))(z.astype(x.dtype))
    return jnp.sum((x_hat - x) ** 2, -1)


def gauss_ll(cov, r):
    """-1/2 (r^T cov^-1 r + log det cov), no 2 pi term: cov (..., N, N), r (..., N, Q) -> (..., Q)."""
    lc = jnp.linalg.cholesky(cov)
    a = jsl.solve_triangular(lc, r, lower=True)
    return -0.5 * (jnp.sum(a ** 2, -2) + 2 * jnp.sum(jnp.log(jnp.diagonal(lc, axis1=-2, axis2=-1)), -1)[..., None])


def restrict(cov, o):
    """Covariance (..., G, G) on the grid points o (..., N, G) of a clip: (..., N, N), identity on padding frames."""
    valid = o.sum(-1)
    return (o @ cov) @ jnp.swapaxes(o, -1, -2) + jnp.eye(N) * (1 - valid)[..., None, :]


def ce_ll(cov, m, y, lengths):
    """Log-likelihood (Q, B) (no 2 pi term) of the codes y (Q, d, G) of Q clips of `lengths` frames under the B signs,
    N(m (B, d, G), cov (B, d, G, G)) restricted to each clip's grid points, summed over the dims. Clips of the same length
    share their grid points: one Cholesky per (length, sign, dim), memory bounded by a checkpointed scan."""
    o = PATTERNS[lengths - MIN_FRAMES]  # (Q, N, G)
    r = jnp.einsum("qkg,qbdg->bdkq", o, y[:, None] - m[None])  # (B, d, N, Q)

    @jax.checkpoint
    def one(out, p):
        ll = gauss_ll(restrict(cov, PATTERNS[p][None, None]), r).sum(1)  # (B, Q)
        return jnp.where((lengths - MIN_FRAMES == p)[:, None], ll.T, out), None
    out, _ = jax.lax.scan(one, jnp.zeros((len(y), len(m)), F64), jnp.arange(len(LENGTHS)))
    return out


def own_ll(cov, m, y, lengths):
    """Log-likelihood (Q,) of each clip under its own (cov (Q, d, G, G), m (Q, d, G)), same convention as ce_ll."""
    o = PATTERNS[lengths - MIN_FRAMES][:, None]  # (Q, 1, N, G)
    r = jnp.einsum("qdkg,qdg->qdk", o, y - m)[..., None]
    return gauss_ll(restrict(cov, o), r)[..., 0].sum(1)


def interp_mask(lengths):
    """Frames seen by the interpolation term: the first two and the last two of each clip, (..., N) bool."""
    k = jnp.arange(N)
    return (k < 2) | ((k >= lengths[..., None] - 2) & (k < lengths[..., None]))


def terms(model, x, idx, lengths, key, beta=BETA, ce_temp=1.0):
    """Loss terms of a batch x (B, n, N, 274), grid indices idx (B, n, N), lengths (B, n): B signs with n clips each.
    beta (2 sigma^2, nats -> rec units) weighs the KL and the CE; interp is a squared error, not weighted by beta.
    ce_temp: CE on the logits / ce_temp, its weight times ce_temp; the returned ce is the tempered one.
    Returns (loss, terms (len(TERMS),))."""
    vae, prior = model.vae, model.prior
    mu, logvar = jax.vmap(jax.vmap(jax.vmap(vae.encode)))(x)  # (B, n, N, d)
    b, n, _, d = mu.shape
    oh = onehot(idx)
    real = (idx >= 0).astype(F64)
    frames = jnp.sum(real)
    m0 = prior.m()
    y, lam = to_grid(m0, mu, logvar, oh)  # (B, d, n, G)
    k_main, k_aux, k_int = jax.random.split(key, 3)
    live = batch_posterior(chols(prior), y, lam, k_main)
    rec = jnp.sum(sq_err(vae, x, to_frames(live["z"], oh) + m0) * real) / frames
    kl_mean, kl_res = jnp.sum(live["kl_mean"]) / frames, jnp.sum(live["kl_res"]) / frames

    # CE, HPs frozen in this term
    l0, ls, k_clip = chols(frozen(prior))
    aux = batch_posterior((l0, ls, k_clip), y, lam, k_aux)
    m_loo, c_loo = jax.vmap(jax.vmap(_loo), (None, 0, 0))(l0, aux["binv"], aux["by"])  # (B, d, n, G(, G))
    s2bar = jnp.sum(jnp.where(lam > 0, 1 / jnp.where(lam > 0, lam, 1.0), 0.0), (0, 2, 3)) / frames
    dbar = jax.lax.stop_gradient(s2bar)[:, None, None] * EYE  # (d, G, G)
    yq = jnp.moveaxis(y, 1, 2).reshape(b * n, d, G)
    lq = lengths.reshape(-1)
    cov = jnp.einsum("bdki,bdkj->bdij", aux["a"], aux["a"]) + k_clip + dbar  # (B, d, G, G)
    ll = ce_ll(cov, aux["m"], yq, lq)  # (B n, B)
    ml = jnp.moveaxis(m_loo, 1, 2).reshape(b * n, d, G)
    cl = jnp.moveaxis(c_loo, 1, 2).reshape(b * n, d, G, G) + k_clip + dbar
    rows, own = jnp.arange(b * n), jnp.repeat(jnp.arange(b), n)
    ll = ll.at[rows, own].set(own_ll(cl, ml, yq, lq)) / ce_temp
    ce = -jnp.mean(jax.nn.log_softmax(ll, -1)[rows, own])

    interp = jnp.zeros((), F64)
    if model.has_interp:
        seen = interp_mask(lengths) * real  # (B, n, N)
        obs = jnp.einsum("bnk,bnkg->bng", seen, oh)[:, None].repeat(d, 1)  # (B, d, n, G)
        p_i = gscale(prior, INTERP_HP)
        m0_i = p_i.m()
        y_i, lam_i = to_grid(m0_i, mu, logvar, oh)
        l0_i, ls_i, _ = chols(p_i)
        binv, by = clip_stats(ls_i, y_i, lam_i)
        zi = jax.vmap(jax.vmap(_interp), (None, None, 0, 0, 0, 0, 0, 0))(l0_i, ls_i, y_i, lam_i, binv, by, obs,
                                                                         jax.random.split(k_int, (b, d)))
        interp = jnp.sum(sq_err(vae, x, to_frames(zi, oh) + m0_i) * (real - seen)) / frames
    loss = rec + (beta * (kl_mean + kl_res + ALPHA * ce_temp * ce * b * n / frames)
                  + INTERP_WEIGHT * interp).astype(rec.dtype)
    return loss, jnp.stack([rec, kl_mean, kl_res, ce, interp])


# ---------------------------------------------------------------- checkpoints
def save(path, model, info):
    """Writes path.eqx (VAE weights) and path.json (info, which must contain "dims", + prior HPs)."""
    eqx.tree_serialise_leaves(f"{path}.eqx", model.vae)
    with open(f"{path}.json", "w") as f:
        json.dump({**info, "prior": to_json(model.prior), "sign_kind": model.prior.sign_kind,
                   "clip_kind": model.prior.clip_kind}, f)


def load(path):
    """Inverse of save: (Model, info)."""
    with open(f"{path}.json") as f:
        info = json.load(f)
    like = PartVAE(info["dims"], tuple(np.zeros(SIZES[p], np.float32) for p in PARTS), jax.random.key(0))
    like = jax.tree.map(lambda a: a.astype(jnp.float32) if eqx.is_inexact_array(a) else a, like)
    vae = eqx.tree_deserialise_leaves(f"{path}.eqx", like)
    return Model(vae, from_json(info["prior"], sum(vae.dims), info["sign_kind"], info["clip_kind"])), info
