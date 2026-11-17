"""
Train a small GPT with hand-written backpropagation, then sample from it.

The model is the one of Andrej Karpathy's microgpt: token and position
embeddings, RMSNorm, causal multi-head attention, a ReLU MLP, residual
connections and no biases.  The forward and backward passes work on a whole
token sequence with NumPy array operations; tokenizing, the Adam loop over
the parameter dictionary and token-by-token sampling are plain Python.

The matrices are small (at most 128x32), so the benchmark measures the cost
of many small NumPy calls and of the Python code between them, rather than
BLAS.
"""

import math
import random

import numpy as np

STEPS = 200
SAMPLES = 16

# Data: pronounceable pseudo-names from a small syllable grammar.
ONSETS = ["", ""] + "b d k l m n r s t v z j h br kr ch sh th".split()
VOWELS = "a a e e i o u y ia ai ei".split()
CODAS = ["", "", ""] + "n l r s m h th nn".split()

N_LAYER, N_EMBD, N_HEAD, BLOCK_SIZE = 2, 32, 4, 16
HEAD_DIM = N_EMBD // N_HEAD
INV_SQRT_HEAD_DIM = 1.0 / math.sqrt(HEAD_DIM)
CAUSAL_MASK = np.triu(np.full((BLOCK_SIZE, BLOCK_SIZE), -np.inf), k=1)
BETA1, BETA2, EPS_ADAM = 0.85, 0.99, 1e-8


def make_names(count, rnd):
    names = set()
    while len(names) < count:
        word = "".join(
            rnd.choice(ONSETS) + rnd.choice(VOWELS) + rnd.choice(CODAS)
            for _ in range(rnd.choice((1, 2, 2, 2, 3, 3)))
        )
        if 2 <= len(word) <= 12:
            names.add(word)
    return sorted(names)


def init_params(vocab_size, rng, std=0.08):
    params = {
        "wte": rng.normal(0, std, (vocab_size, N_EMBD)),
        "wpe": rng.normal(0, std, (BLOCK_SIZE, N_EMBD)),
        "lm_head": rng.normal(0, std, (N_EMBD, vocab_size)),
    }
    for li in range(N_LAYER):
        for name in ("wq", "wk", "wv", "wo"):
            params[f"l{li}.{name}"] = rng.normal(0, std, (N_EMBD, N_EMBD))
        params[f"l{li}.fc1"] = rng.normal(0, std, (N_EMBD, 4 * N_EMBD))
        params[f"l{li}.fc2"] = rng.normal(0, std, (4 * N_EMBD, N_EMBD))
    return params


def rmsnorm(x):
    scale = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-5)
    return x * scale, scale


def rmsnorm_backward(dy, x, scale):
    return dy * scale - x * scale**3 * (dy * x).mean(axis=-1, keepdims=True)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def forward(params, idx):
    """Logits for every position of the token list, and what backward needs."""
    seq_len = len(idx)
    emb = params["wte"][idx] + params["wpe"][:seq_len]
    x, emb_scale = rmsnorm(emb)
    mask = CAUSAL_MASK[:seq_len, :seq_len]
    layers = []
    for li in range(N_LAYER):
        # 1) causal multi-head self-attention
        xn, scale1 = rmsnorm(x)
        q = xn @ params[f"l{li}.wq"]
        k = xn @ params[f"l{li}.wk"]
        v = xn @ params[f"l{li}.wv"]
        y = np.empty_like(x)
        weights = []
        for h in range(N_HEAD):
            s = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            w = softmax(q[:, s] @ k[:, s].T * INV_SQRT_HEAD_DIM + mask)
            y[:, s] = w @ v[:, s]
            weights.append(w)
        x_mid = x + y @ params[f"l{li}.wo"]
        # 2) MLP
        xm, scale2 = rmsnorm(x_mid)
        r = np.maximum(xm @ params[f"l{li}.fc1"], 0.0)
        x_out = x_mid + r @ params[f"l{li}.fc2"]
        layers.append((x, xn, scale1, q, k, v, weights, y, x_mid, xm, scale2, r))
        x = x_out
    return x @ params["lm_head"], (emb, emb_scale, layers, x)


def loss_and_grads(params, idx, targets):
    seq_len = len(idx)
    logits, (emb, emb_scale, layers, x) = forward(params, idx)
    probs = softmax(logits)
    rows = np.arange(seq_len)
    loss = float(-np.log(probs[rows, targets]).mean())

    grads = {}
    dlogits = probs
    dlogits[rows, targets] -= 1.0
    dlogits /= seq_len
    grads["lm_head"] = x.T @ dlogits
    dx = dlogits @ params["lm_head"].T
    for li in reversed(range(N_LAYER)):
        x_in, xn, scale1, q, k, v, weights, y, x_mid, xm, scale2, r = layers[li]
        # MLP
        grads[f"l{li}.fc2"] = r.T @ dx
        dh = (dx @ params[f"l{li}.fc2"].T) * (r > 0.0)
        grads[f"l{li}.fc1"] = xm.T @ dh
        dx = dx + rmsnorm_backward(dh @ params[f"l{li}.fc1"].T, x_mid, scale2)
        # attention
        grads[f"l{li}.wo"] = y.T @ dx
        dy = dx @ params[f"l{li}.wo"].T
        dq, dk, dv = np.empty_like(q), np.empty_like(k), np.empty_like(v)
        for h in range(N_HEAD):
            s = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            w = weights[h]
            dw = dy[:, s] @ v[:, s].T
            dv[:, s] = w.T @ dy[:, s]
            datt = w * (dw - (dw * w).sum(axis=-1, keepdims=True))
            dq[:, s] = datt @ k[:, s] * INV_SQRT_HEAD_DIM
            dk[:, s] = datt.T @ q[:, s] * INV_SQRT_HEAD_DIM
        grads[f"l{li}.wq"] = xn.T @ dq
        grads[f"l{li}.wk"] = xn.T @ dk
        grads[f"l{li}.wv"] = xn.T @ dv
        dxn = (
            dq @ params[f"l{li}.wq"].T
            + dk @ params[f"l{li}.wk"].T
            + dv @ params[f"l{li}.wv"].T
        )
        dx = dx + rmsnorm_backward(dxn, x_in, scale1)
    demb = rmsnorm_backward(dx, emb, emb_scale)
    grads["wte"] = np.zeros_like(params["wte"])
    np.add.at(grads["wte"], idx, demb)
    grads["wpe"] = np.zeros_like(params["wpe"])
    grads["wpe"][:seq_len] = demb
    return loss, grads


def adam_step(params, grads, m_state, v_state, lr, t):
    c1 = 1.0 - BETA1**t
    c2 = 1.0 - BETA2**t
    for name, g in grads.items():
        m, v = m_state[name], v_state[name]
        m *= BETA1
        m += (1.0 - BETA1) * g
        v *= BETA2
        v += (1.0 - BETA2) * g * g
        params[name] -= lr * (m / c1) / (np.sqrt(v / c2) + EPS_ADAM)


def setup():
    rnd = random.Random(42)
    docs = make_names(600, rnd)
    rnd.shuffle(docs)
    return docs, sorted(set("".join(docs)))


def bench(state):
    docs, chars = state
    rnd = random.Random(1)
    rng = np.random.default_rng(1)
    stoi = {ch: i for i, ch in enumerate(chars)}
    bos = len(chars)
    vocab_size = bos + 1
    params = init_params(vocab_size, rng)
    m_state = {name: np.zeros_like(w) for name, w in params.items()}
    v_state = {name: np.zeros_like(w) for name, w in params.items()}

    # Train on one document per step.
    losses = []
    for step in range(STEPS):
        doc = docs[step % len(docs)]
        tokens = [bos] + [stoi[ch] for ch in doc] + [bos]
        loss, grads = loss_and_grads(params, tokens[:-1], tokens[1:])
        lr = 0.01 * (1.0 - step / STEPS)
        adam_step(params, grads, m_state, v_state, lr, step + 1)
        losses.append(loss)

    # Sample new names token by token.
    samples = []
    for _ in range(SAMPLES):
        tokens = [bos]
        while len(tokens) < BLOCK_SIZE:
            logits = forward(params, tokens)[0][-1]
            probs = softmax(logits / 0.6).tolist()
            token = rnd.choices(range(vocab_size), weights=probs)[0]
            if token == bos:
                break
            tokens.append(token)
        samples.append("".join(chars[t] for t in tokens[1:]))
    return losses, samples


def verify(result):
    losses, samples = result
    first = sum(losses[:20]) / 20
    last = sum(losses[-20:]) / 20
    if not (len(losses) == STEPS and last < first):
        raise RuntimeError(f"training did not reduce the loss: {first} -> {last}")
    if len(samples) != SAMPLES:
        raise RuntimeError(f"expected {SAMPLES} samples, got {len(samples)}")
