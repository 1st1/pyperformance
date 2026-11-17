"""
Fit Gaussian mixtures with EM after reducing the data with PCA.

Points drawn from a few Gaussian clusters in a low-dimensional subspace are
embedded in a higher-dimensional space with noise.  The pipeline recovers
the subspace with PCA (power iteration with deflation), then fits
full-covariance Gaussian mixtures by expectation maximization for a range
of component counts, each started from k-means++ and Lloyd's algorithm, and
picks the count by the Bayesian information criterion.

Almost all the work is whole-array operations on a few thousand points;
Python drives the loops over components and candidate models.  Of the
benchmarks in this group it spends the largest share of its time inside
NumPy.
"""

import math

import numpy as np

N_POINTS, TRUE_K, LATENT, AMBIENT = 4000, 5, 4, 24
CANDIDATES = (3, 4, 5, 6)
LOG_2PI = math.log(2.0 * math.pi)


def make_data(rng):
    means = rng.normal(0.0, 4.0, (TRUE_K, LATENT))
    labels = rng.integers(0, TRUE_K, N_POINTS)
    z = np.empty((N_POINTS, LATENT))
    for j in range(TRUE_K):
        m = rng.normal(0.0, 1.0, (LATENT, LATENT))
        cov = m @ m.T / LATENT + 0.3 * np.eye(LATENT)
        members = labels == j
        z[members] = rng.multivariate_normal(means[j], cov, int(members.sum()))
    basis, _ = np.linalg.qr(rng.normal(0.0, 1.0, (AMBIENT, LATENT)))
    return z @ basis.T + rng.normal(0.0, 0.3, (N_POINTS, AMBIENT)), labels


def pca_power(x, d, rng, max_iter=500, tol=1e-12):
    """The top d principal components by power iteration with deflation."""
    xc = x - x.mean(axis=0)
    c = xc.T @ xc / (len(x) - 1)
    components = []
    for _ in range(d):
        v = rng.normal(0.0, 1.0, c.shape[0])
        v /= np.linalg.norm(v)
        lam = 0.0
        for _ in range(max_iter):
            w = c @ v
            new_lam = float(v @ w)
            v = w / np.linalg.norm(w)
            if abs(new_lam - lam) <= tol * abs(new_lam):
                break
            lam = new_lam
        components.append(v)
        c = c - new_lam * np.outer(v, v)
    return xc @ np.array(components).T


def squared_distances(z, centers):
    return (
        (z * z).sum(axis=1)[:, None]
        - 2.0 * (z @ centers.T)
        + (centers * centers).sum(axis=1)
    )


def kmeans(z, k, rng, max_iter=100):
    # k-means++ seeding
    centers = [z[rng.integers(len(z))]]
    for _ in range(k - 1):
        d2 = squared_distances(z, np.array(centers)).min(axis=1).clip(min=0.0)
        centers.append(z[rng.choice(len(z), p=d2 / d2.sum())])
    centers = np.array(centers)
    # Lloyd iterations
    labels = None
    for _ in range(max_iter):
        new_labels = squared_distances(z, centers).argmin(axis=1)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = labels == j
            if members.any():
                centers[j] = z[members].mean(axis=0)
    return centers, labels


def log_gaussian(z, mean, cov):
    d = z.shape[1]
    chol = np.linalg.cholesky(cov)
    y = np.linalg.solve(chol, (z - mean).T)
    log_det = float(np.log(np.diag(chol)).sum())
    return -0.5 * ((y * y).sum(axis=0) + d * LOG_2PI) - log_det


def fit_gmm(z, k, rng, max_iter=40, tol=1e-6, reg=1e-6):
    n, d = z.shape
    means, labels = kmeans(z, k, rng)
    weights = np.bincount(labels, minlength=k) / n
    eye = np.eye(d)
    covs = []
    for j in range(k):
        members = z[labels == j]
        # A cluster too small for its own covariance starts from the global one.
        covs.append(np.cov(members.T) + reg * eye if len(members) > d else np.cov(z.T))
    log_p = np.empty((n, k))
    previous = -math.inf
    for _ in range(max_iter):
        # E step: responsibilities via log-sum-exp
        for j in range(k):
            log_p[:, j] = math.log(weights[j]) + log_gaussian(z, means[j], covs[j])
        top = log_p.max(axis=1, keepdims=True)
        log_norm = top[:, 0] + np.log(np.exp(log_p - top).sum(axis=1))
        ll = float(log_norm.sum())
        resp = np.exp(log_p - log_norm[:, None])
        # M step
        nk = resp.sum(axis=0)
        weights = nk / n
        means = (resp.T @ z) / nk[:, None]
        for j in range(k):
            zc = z - means[j]
            covs[j] = (zc * resp[:, j : j + 1]).T @ zc / nk[j] + reg * eye
        if ll - previous <= tol * abs(ll):
            break
        previous = ll
    n_params = (k - 1) + k * d + k * d * (d + 1) // 2
    return resp, ll, -2.0 * ll + n_params * math.log(n)


def purity(assigned, truth, k):
    """The share of points whose component's majority label is their own."""
    correct = 0
    for j in range(k):
        members = truth[assigned == j]
        if len(members):
            correct += int(np.bincount(members).max())
    return correct / len(truth)


def setup():
    return make_data(np.random.default_rng(5))


def bench(state):
    x, truth = state
    rng = np.random.default_rng(9)
    z = pca_power(x, LATENT, rng)
    best = None
    for k in CANDIDATES:
        resp, ll, bic = fit_gmm(z, k, rng)
        if best is None or bic < best[2]:
            best = (k, resp, bic)
    k, resp, _ = best
    return k, purity(resp.argmax(axis=1), truth, k)


def verify(result):
    k, score = result
    if k not in CANDIDATES or not score > 0.8:
        raise RuntimeError(f"poor fit: {k} components with purity {score}")
