"""
Train a multi-layer perceptron classifier with manual backpropagation.

Three interleaved, noisy spiral arms are separated by a small ReLU network
built from layer objects (Dense, ReLU), a softmax cross-entropy loss and an
Adam optimizer with a warm-up and cosine learning-rate schedule.  Training
is mini-batch: matrix products and elementwise array operations per batch,
driven by Python loops over batches, layers and parameters.
"""

import math

import numpy as np

EPOCHS, BATCH_SIZE = 12, 32
BASE_LR, WARMUP = 1e-2, 50


def make_spirals(per_class, classes, rng, noise=0.25, turns=1.5):
    x = np.empty((per_class * classes, 2))
    y = np.repeat(np.arange(classes), per_class)
    for c in range(classes):
        r = np.linspace(0.05, 1.0, per_class)
        theta = 2.0 * np.pi * (c / classes + turns * r)
        theta += rng.normal(0.0, noise, per_class)
        rows = slice(c * per_class, (c + 1) * per_class)
        x[rows] = np.column_stack([r * np.sin(theta), r * np.cos(theta)])
    return x, y


class Dense:
    def __init__(self, n_in, n_out, rng):
        self.w = rng.normal(0.0, math.sqrt(2.0 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, dy):
        np.matmul(self.x.T, dy, out=self.dw)
        np.sum(dy, axis=0, out=self.db)
        return dy @ self.w.T

    def parameters(self):
        return [(self.w, self.dw), (self.b, self.db)]


class ReLU:
    def forward(self, x):
        self.mask = x > 0.0
        return x * self.mask

    def backward(self, dy):
        return dy * self.mask

    def parameters(self):
        return []


class Network:
    def __init__(self, sizes, rng):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Dense(sizes[i], sizes[i + 1], rng))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        return [pair for layer in self.layers for pair in layer.parameters()]


def softmax_cross_entropy(logits, y):
    """Mean loss and its gradient with respect to the logits."""
    z = logits - logits.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    rows = np.arange(len(y))
    loss = float(-log_probs[rows, y].mean())
    grad = np.exp(log_probs)
    grad[rows, y] -= 1.0
    grad /= len(y)
    return loss, grad


class Adam:
    def __init__(self, parameters, beta1=0.9, beta2=0.999, eps=1e-8, decay=1e-4):
        self.parameters = parameters
        self.beta1, self.beta2, self.eps, self.decay = beta1, beta2, eps, decay
        self.m = [np.zeros_like(p) for p, _ in parameters]
        self.v = [np.zeros_like(p) for p, _ in parameters]
        self.t = 0

    def step(self, lr):
        self.t += 1
        beta1, beta2 = self.beta1, self.beta2
        step_size = lr * math.sqrt(1.0 - beta2**self.t) / (1.0 - beta1**self.t)
        for (p, g), m, v in zip(self.parameters, self.m, self.v):
            if p.ndim == 2:
                g = g + self.decay * p
            m *= beta1
            m += (1.0 - beta1) * g
            v *= beta2
            v += (1.0 - beta2) * (g * g)
            p -= step_size * m / (np.sqrt(v) + self.eps)


def accuracy(net, x, y):
    return float((net.forward(x).argmax(axis=1) == y).mean())


def setup():
    rng = np.random.default_rng(3)
    x, y = make_spirals(600, 3, rng)
    order = rng.permutation(len(x))
    split = len(x) * 4 // 5
    train, test = order[:split], order[split:]
    mean, std = x[train].mean(axis=0), x[train].std(axis=0)
    return (x[train] - mean) / std, y[train], (x[test] - mean) / std, y[test]


def bench(state):
    x_train, y_train, x_test, y_test = state
    rng = np.random.default_rng(5)
    net = Network([2, 64, 64, 64, 3], rng)
    opt = Adam(net.parameters())
    n = len(x_train)
    total_steps = EPOCHS * math.ceil(n / BATCH_SIZE)

    step = 0
    history = []
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        epoch_loss, batches = 0.0, 0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            if step < WARMUP:
                lr = BASE_LR * (step + 1) / WARMUP
            else:
                progress = (step - WARMUP) / (total_steps - WARMUP)
                lr = BASE_LR * 0.5 * (1.0 + math.cos(math.pi * progress))
            logits = net.forward(x_train[idx])
            loss, grad = softmax_cross_entropy(logits, y_train[idx])
            net.backward(grad)
            opt.step(lr)
            epoch_loss += loss
            batches += 1
            step += 1
        history.append((epoch_loss / batches, accuracy(net, x_test, y_test)))
    return history


def verify(history):
    first_loss, last_loss = history[0][0], history[-1][0]
    if not (len(history) == EPOCHS and last_loss < first_loss):
        raise RuntimeError(
            f"training did not reduce the loss: {first_loss} -> {last_loss}"
        )
    if not history[-1][1] > 0.8:
        raise RuntimeError(f"test accuracy {history[-1][1]} is too low")
