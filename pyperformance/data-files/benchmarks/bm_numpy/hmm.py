"""
Fit a discrete hidden Markov model with Baum-Welch, then decode with Viterbi.

Sequences of varying length are drawn from a known model.  Expectation
maximization runs the scaled forward-backward recursions one sequence at a
time: a Python loop over time steps around small vector-matrix products,
with the expected counts accumulated by array operations.  The fitted model
then decodes a second set of sequences, sampled in pure Python, with the
Viterbi algorithm, and the decoded states are scored against the true ones.
"""

import random
from bisect import bisect
from itertools import accumulate

import numpy as np

N_STATES, N_SYMBOLS = 5, 12
N_SEQUENCES = 16
EM_ITERATIONS = 4


def make_model(rng, stickiness=0.9):
    pi = rng.dirichlet(np.full(N_STATES, 2.0))
    a = stickiness * np.eye(N_STATES) + (1.0 - stickiness) * rng.dirichlet(
        np.full(N_STATES, 1.0), N_STATES
    )
    b = rng.dirichlet(np.full(N_SYMBOLS, 0.2), N_STATES)
    return pi, a, b


def sample_sequences(model, lengths, rnd):
    """Draw (states, observations) pairs with inverse-CDF sampling."""
    pi, a, b = model
    last_state, last_symbol = N_STATES - 1, N_SYMBOLS - 1
    cum_pi = list(accumulate(pi.tolist()))
    cum_a = [list(accumulate(row)) for row in a.tolist()]
    cum_b = [list(accumulate(row)) for row in b.tolist()]
    out = []
    for length in lengths:
        state = min(bisect(cum_pi, rnd.random()), last_state)
        states, obs = [], []
        for _ in range(length):
            states.append(state)
            obs.append(min(bisect(cum_b[state], rnd.random()), last_symbol))
            state = min(bisect(cum_a[state], rnd.random()), last_state)
        out.append((states, obs))
    return out


def forward_backward(obs, pi, a, b):
    """Scaled recursions.

    Returns the state posteriors, the summed transition posteriors and the
    log-likelihood of the sequence.
    """
    n = len(obs)
    emit = b[:, obs].T  # (n, K): probability of each observation per state
    alpha = np.empty_like(emit)
    c = np.empty(n)
    v = pi * emit[0]
    c[0] = v.sum()
    alpha[0] = v / c[0]
    for t in range(1, n):
        v = (alpha[t - 1] @ a) * emit[t]
        c[t] = s = v.sum()
        alpha[t] = v / s
    beta = np.empty_like(emit)
    beta[n - 1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (a @ (emit[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    xi = a * (alpha[:-1].T @ (emit[1:] * beta[1:] / c[1:, None]))
    return gamma, xi, float(np.log(c).sum())


def baum_welch(sequences, rng, iterations):
    # Start from a guess that states persist; a flat start converges slowly.
    pi = np.full(N_STATES, 1.0 / N_STATES)
    a = 0.9 * np.eye(N_STATES) + 0.1 * rng.dirichlet(np.ones(N_STATES), N_STATES)
    b = rng.dirichlet(np.ones(N_SYMBOLS), N_STATES)
    history = []
    for _ in range(iterations):
        pi_acc = np.zeros(N_STATES)
        a_acc = np.zeros((N_STATES, N_STATES))
        b_acc = np.zeros((N_SYMBOLS, N_STATES))
        total = 0.0
        for obs in sequences:
            gamma, xi, ll = forward_backward(obs, pi, a, b)
            pi_acc += gamma[0]
            a_acc += xi
            np.add.at(b_acc, obs, gamma)
            total += ll
        history.append(total)
        pi = pi_acc / pi_acc.sum()
        a = a_acc / a_acc.sum(axis=1, keepdims=True)
        b = (b_acc / b_acc.sum(axis=0)).T
    return (pi, a, b), history


def viterbi(obs, log_pi, log_a, log_b):
    n = len(obs)
    emit = log_b[:, obs].T
    back = np.empty((n, N_STATES), dtype=np.intp)
    delta = log_pi + emit[0]
    for t in range(1, n):
        # scores[i, j]: the best path ending in state i, then i -> j
        scores = delta[:, None] + log_a
        back[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + emit[t]
    state = int(delta.argmax())
    path = [state]
    for t in range(n - 1, 0, -1):
        state = int(back[t, state])
        path.append(state)
    path.reverse()
    return path


def match_states(pairs):
    """Greedy one-to-one map from decoded to true states by co-occurrence."""
    counts = {}
    for truth, decoded in pairs:
        for key in zip(decoded, truth):
            counts[key] = counts.get(key, 0) + 1
    mapping, taken = {}, set()
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    for (decoded, truth), _ in ranked:
        if decoded not in mapping and truth not in taken:
            mapping[decoded] = truth
            taken.add(truth)
    for decoded in range(N_STATES):
        mapping.setdefault(decoded, -1)
    return mapping


def setup():
    rnd = random.Random(7)
    rng = np.random.default_rng(7)
    model = make_model(rng)
    lengths = [rnd.randint(150, 350) for _ in range(N_SEQUENCES)]
    train = [np.array(obs) for _, obs in sample_sequences(model, lengths, rnd)]
    return model, train


def bench(state):
    model, train = state
    rnd = random.Random(11)
    rng = np.random.default_rng(11)

    fitted, history = baum_welch(train, rng, EM_ITERATIONS)

    # Decode sequences the model has not seen.
    lengths = [rnd.randint(150, 350) for _ in range(N_SEQUENCES)]
    test = sample_sequences(model, lengths, rnd)
    with np.errstate(divide="ignore"):
        log_pi, log_a, log_b = (np.log(m) for m in fitted)
    pairs = [
        (states, viterbi(np.array(obs), log_pi, log_a, log_b)) for states, obs in test
    ]
    mapping = match_states(pairs)
    correct = sum(
        1 for truth, decoded in pairs for s, d in zip(truth, decoded) if mapping[d] == s
    )
    return history, correct / sum(lengths)


def verify(result):
    history, accuracy = result
    for before, after in zip(history, history[1:]):
        if after < before - 1e-6 * abs(after):
            raise RuntimeError(f"log-likelihood fell: {before} -> {after}")
    if not accuracy > 1.5 / N_STATES:
        raise RuntimeError(f"decoded accuracy {accuracy} is close to chance")
