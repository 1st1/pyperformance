"""
NumPy benchmarks.

Five well-known algorithms, written the way NumPy code usually is: array
operations for the numerical work and plain Python for the control flow
around them.  They differ in how the time divides between the interpreter,
the fixed cost of calling into NumPy on small arrays, and work on large
arrays:

* gpt: train a small transformer with hand-written backpropagation
* hmm: Baum-Welch training and Viterbi decoding of a hidden Markov model
* kalman: a Kalman filter and RTS smoother on small matrices
* mlp: train a multi-layer perceptron built from layer objects
* gmm: PCA followed by Gaussian mixture fitting on a few thousand points

Each algorithm lives in its own module with three functions: setup() builds
the input, bench(state) is the timed workload and verify(result) checks that
the workload did what it is meant to do.
"""

import importlib

import pyperf

BENCHMARKS = ("gmm", "gpt", "hmm", "kalman", "mlp")


def add_cmdline_args(cmd, args):
    cmd.append(args.benchmark)


def add_parser_args(parser):
    parser.add_argument(
        "benchmark",
        choices=BENCHMARKS,
        help="Which benchmark to run.",
    )


def main():
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata["description"] = "NumPy benchmarks"
    add_parser_args(runner.argparser)
    args = runner.parse_args()
    module = importlib.import_module(args.benchmark)

    state = None
    if args.worker:
        # Only the worker processes run the workload.  Check its result once,
        # outside the measurement.
        state = module.setup()
        module.verify(module.bench(state))

    runner.bench_func(f"numpy_{args.benchmark}", module.bench, state)


if __name__ == "__main__":
    main()
