"""
Track maneuvering targets with a Kalman filter and an RTS smoother.

Each target follows a constant-velocity model in two dimensions driven by
random accelerations.  Its position is measured with noise; some
measurements are missing and some are gross outliers.  The filter predicts
and updates one time step at a time, gates outliers on the normalized
innovation, re-acquires a track after several rejections in a row, and a
Rauch-Tung-Striebel backward pass smooths the estimates.

Every step is a dozen products of 4x4 and 4x2 matrices and a 2x2 inverse,
driven by a Python loop: the common shape of NumPy code that cannot be
vectorized over time.
"""

import math

import numpy as np

TARGETS, STEPS = 6, 500
DT, SIGMA_A, SIGMA_Z = 1.0, 0.05, 2.0
P_MISS, P_OUTLIER, OUTLIER_SPREAD = 0.08, 0.03, 40.0
GATE = 13.82  # chi-square, 2 degrees of freedom, 99.9%
LOST_AFTER = 3  # rejections in a row before a track counts as lost

F = np.array(
    [
        [1.0, 0.0, DT, 0.0],
        [0.0, 1.0, 0.0, DT],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
G = np.array([[DT * DT / 2, 0.0], [0.0, DT * DT / 2], [DT, 0.0], [0.0, DT]])
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
Q = G @ G.T * SIGMA_A**2
R = np.eye(2) * SIGMA_Z**2
I4 = np.eye(4)
P_LOST = np.diag([1e4, 1e4, 25.0, 25.0])


def simulate(steps, rng):
    """True states, measurements, and flags for present and outlier ones."""
    accel = rng.normal(0.0, SIGMA_A, (steps, 2))
    noise = rng.normal(0.0, SIGMA_Z, (steps, 2))
    u = rng.random(steps)
    seen = u >= P_MISS
    outlier = seen & (u < P_MISS + P_OUTLIER)
    noise[outlier] += rng.normal(0.0, OUTLIER_SPREAD, (int(outlier.sum()), 2))
    truth = np.empty((steps, 4))
    x = np.concatenate([rng.uniform(-100.0, 100.0, 2), rng.normal(0.0, 1.0, 2)])
    for t in range(steps):
        x = F @ x + G @ accel[t]
        truth[t] = x
    return truth, truth[:, :2] + noise, seen, outlier


def kalman_filter(zs, seen, x, p):
    n = len(zs)
    x_pred, p_pred = np.empty((n, 4)), np.empty((n, 4, 4))
    x_filt, p_filt = np.empty((n, 4)), np.empty((n, 4, 4))
    rejected, streak = 0, 0
    for t in range(n):
        # predict
        x = F @ x
        p = F @ p @ F.T + Q
        x_pred[t], p_pred[t] = x, p
        # update, unless the measurement is missing or fails the gate
        if seen[t]:
            y = zs[t] - H @ x
            s_inv = np.linalg.inv(H @ p @ H.T + R)
            accept = float(y @ s_inv @ y) < GATE
            if accept:
                streak = 0
            else:
                rejected += 1
                streak += 1
                if streak == LOST_AFTER:
                    # The track is lost, not the measurements: widen the
                    # uncertainty and start over from this one.
                    p = p + P_LOST
                    s_inv = np.linalg.inv(H @ p @ H.T + R)
                    accept, streak = True, 0
            if accept:
                k = p @ H.T @ s_inv
                x = x + k @ y
                ikh = I4 - k @ H
                # The Joseph form keeps the covariance symmetric.
                p = ikh @ p @ ikh.T + k @ R @ k.T
        x_filt[t], p_filt[t] = x, p
    return x_pred, p_pred, x_filt, p_filt, rejected


def rts_smoother(x_pred, p_pred, x_filt, p_filt):
    x_smooth, p_smooth = x_filt.copy(), p_filt.copy()
    for t in range(len(x_filt) - 2, -1, -1):
        c = p_filt[t] @ F.T @ np.linalg.inv(p_pred[t + 1])
        x_smooth[t] = x_filt[t] + c @ (x_smooth[t + 1] - x_pred[t + 1])
        p_smooth[t] = p_filt[t] + c @ (p_smooth[t + 1] - p_pred[t + 1]) @ c.T
    return x_smooth, p_smooth


def setup():
    rng = np.random.default_rng(11)
    return [simulate(STEPS, rng) for _ in range(TARGETS)]


def bench(tracks):
    squares = {"raw": 0.0, "filtered": 0.0, "smoothed": 0.0}
    n_raw = n_all = rejected = 0
    for truth, zs, seen, outlier in tracks:
        # Start at the first measurement that exists, with unknown velocity.
        first = zs[int(seen.argmax())]
        x0 = np.array([first[0], first[1], 0.0, 0.0])
        p0 = np.diag([4.0 * SIGMA_Z**2, 4.0 * SIGMA_Z**2, 4.0, 4.0])
        x_pred, p_pred, x_filt, p_filt, n_rejected = kalman_filter(zs, seen, x0, p0)
        x_smooth, _ = rts_smoother(x_pred, p_pred, x_filt, p_filt)

        clean = seen & ~outlier
        position = truth[:, :2]
        squares["raw"] += float(((zs[clean] - position[clean]) ** 2).sum())
        squares["filtered"] += float(((x_filt[:, :2] - position) ** 2).sum())
        squares["smoothed"] += float(((x_smooth[:, :2] - position) ** 2).sum())
        n_raw += int(clean.sum())
        n_all += len(zs)
        rejected += n_rejected
    return {
        name: math.sqrt(total / (n_raw if name == "raw" else n_all))
        for name, total in squares.items()
    }, rejected


def verify(result):
    rmse, rejected = result
    if not rmse["smoothed"] < rmse["filtered"] < rmse["raw"]:
        raise RuntimeError(f"expected smoothed < filtered < raw errors, got {rmse}")
    if not rejected > 0:
        raise RuntimeError("the gate rejected no measurement")
