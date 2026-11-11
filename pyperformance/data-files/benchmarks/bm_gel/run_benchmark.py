"""
Benchmarks for the query language and schema compilers of Gel
(formerly EdgeDB), a database built on top of PostgreSQL.

Most of Gel is written in Python.  The benchmarks here use a subset of
the Gel server codebase, namely the compiler pipeline that turns the Gel
schema definition language (SDL) into a schema, and EdgeQL queries into
SQL.  No database is needed to run them: nothing ever gets executed.

``gel_schema``
    Compile ``data/schema.esdl``, a ~480 line long schema of an issue
    tracker application: parse the SDL, order the declarations by their
    dependencies and apply them one by one to the standard library schema
    to produce a new schema.  This involves compiling every expression
    in the schema (computed pointers, defaults, constraints, indexes,
    access policies, triggers, etc) to an intermediate representation, and
    some of them to SQL.

``gel_edgeql``
    Compile the queries in ``data/queries.edgeql`` (a large SELECT with
    a deep shape, an INSERT and an UPDATE) against the above schema to SQL:
    parse EdgeQL, compile the AST into the IR (applying access policies,
    rewrites and triggers), compile the IR into a SQL AST and generate
    the SQL text.

See ``vendor/README.rst`` for the details on the vendored Gel code.

Both benchmarks need the schema of the Gel standard library, and it takes
a while to build one (~10 seconds), so that is done once by the main
process which then hands it over to the worker processes as a pickle.
"""

import argparse
import os.path
import pickle
import random
import shutil
import sys
import tempfile
import uuid

import pyperf


BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BENCH_DIR, "data")

sys.path.insert(0, os.path.join(BENCH_DIR, "vendor"))

from edb.common import uuidgen  # noqa: E402
from edb.edgeql import compiler as qlcompiler  # noqa: E402
from edb.edgeql import parser as qlparser  # noqa: E402
from edb.pgsql import codegen as pgcodegen  # noqa: E402
from edb.pgsql import compiler as pgcompiler  # noqa: E402
from edb.schema import ddl as s_ddl  # noqa: E402
from edb.schema import schema as s_schema  # noqa: E402
from edb.schema import std as s_std  # noqa: E402


class UUIDGenerator:
    """A reproducible replacement for edb.common.uuidgen.uuid1mc().

    Gel assigns a UUID to every schema object and uses them as keys in
    the (hash-trie based) schema mappings, as well as to name things in
    the generated SQL.  Make them the same on every run, so that every run
    does the same work.
    """

    def __init__(self):
        self.seed(0)

    def seed(self, seed):
        self._random = random.Random(seed)
        # 100ns intervals since the UUID epoch, like in uuid.uuid1().
        self._timestamp = 0x1F0_0000_0000_0000 + (seed << 40)

    def __call__(self):
        self._timestamp += self._random.randrange(1, 1000)
        timestamp = self._timestamp
        fields = (
            timestamp & 0xFFFFFFFF,
            (timestamp >> 32) & 0xFFFF,
            (timestamp >> 48) & 0x0FFF,
            self._random.getrandbits(6) | 0x80,
            self._random.getrandbits(8),
            # A random multicast "MAC address"
            self._random.getrandbits(48) | (1 << 40),
        )
        return uuidgen.UUID(uuid.UUID(fields=fields, version=1).bytes)


uuidgen.uuid1mc = uuid_generator = UUIDGenerator()


def read_data(filename):
    with open(os.path.join(DATA_DIR, filename), encoding="utf-8") as f:
        return f.read()


def build_std_schema():
    """Compile the standard library of Gel."""
    schema = s_schema.EMPTY_SCHEMA
    for modname in s_schema.STD_SOURCES:
        schema = s_std.load_std_module(schema, modname)
    schema, _ = s_std.make_schema_version(schema)
    schema, _ = s_std.make_global_schema_version(schema)
    return schema


def compile_schema(std_schema, sdl):
    """Compile a schema written in SDL."""
    sdl_ast = qlparser.parse_sdl(sdl)
    schema, _warnings = s_ddl.apply_sdl(sdl_ast, base_schema=std_schema)
    return schema


def compile_queries(schema, edgeql):
    """Compile a script made of EdgeQL queries into a list of SQL queries."""
    options = qlcompiler.CompilerOptions(
        modaliases={None: "default"},
    )
    result = []
    for ql_ast in qlparser.parse_block(edgeql):
        ir = qlcompiler.compile_ast_to_ir(ql_ast, schema, options=options)
        sql_res = pgcompiler.compile_ir_to_sql_tree(
            ir,
            output_format=pgcompiler.OutputFormat.NATIVE,
        )
        result.append(pgcodegen.generate_source(sql_res.ast))
    return result


def bench_schema(loops, state):
    std_schema = state["std_schema"]
    sdl = state["sdl"]
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        compile_schema(std_schema, sdl)

    return pyperf.perf_counter() - t0


def bench_edgeql(loops, state):
    schema = state["schema"]
    edgeql = state["edgeql"]
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        compile_queries(schema, edgeql)

    return pyperf.perf_counter() - t0


BENCHMARKS = {
    # name: (function, default pyperf.Runner() options)
    #
    # Compiling the schema takes a few seconds, so do it only once
    # per worker process, on top of the warmup.
    "schema": (bench_schema, {"values": 1}),
    "edgeql": (bench_edgeql, {"values": 2}),
}


def build_state(benchmark):
    """Build everything needed to run the benchmark."""
    uuid_generator.seed(1)
    state = {"std_schema": build_std_schema()}
    if benchmark == "schema":
        state["sdl"] = read_data("schema.esdl")
    elif benchmark == "edgeql":
        uuid_generator.seed(2)
        state["schema"] = compile_schema(
            state.pop("std_schema"), read_data("schema.esdl")
        )
        state["edgeql"] = read_data("queries.edgeql")
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")
    return state


def get_state(args):
    if args.state_file:
        with open(args.state_file, "rb") as f:
            state = pickle.load(f)
    else:
        state = build_state(args.benchmark)

    # Start every worker off the same state.
    uuid_generator.seed(3)
    random.seed(3)

    # Load the grammar, and warm up the caches and lazy imports.
    compile_queries(state.get("schema") or state["std_schema"], "select 1;")

    return state


def add_cmdline_args(cmd, args):
    if args.state_file:
        cmd.extend(("--state-file", args.state_file))
    cmd.append(args.benchmark)


def add_parser_args(parser):
    parser.add_argument(
        "--state-file",
        metavar="FILE",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "benchmark",
        choices=sorted(BENCHMARKS),
        help="Which benchmark to run.",
    )


def main():
    # The defaults of the runner depend on the benchmark, so find out which
    # one it is going to be before pyperf.Runner() gets to parse the args.
    runner_options = {}
    for arg in sys.argv[1:]:
        if arg in BENCHMARKS:
            _, runner_options = BENCHMARKS[arg]

    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args, **runner_options)
    runner.metadata["description"] = "Gel schema and EdgeQL compilers"
    add_parser_args(runner.argparser)
    args = runner.parse_args()
    bench_func, _ = BENCHMARKS[args.benchmark]
    bench_name = f"gel_{args.benchmark}"

    if args.worker:
        state = get_state(args)
        runner.bench_time_func(bench_name, bench_func, state)
        return

    # This is the main process: prepare the state for the workers.
    tmpdir = tempfile.mkdtemp(prefix="pyperformance-bm_gel-")
    try:
        args.state_file = os.path.join(tmpdir, "state.pickle")
        with open(args.state_file, "wb") as f:
            pickle.dump(build_state(args.benchmark), f, pickle.HIGHEST_PROTOCOL)
        runner.bench_time_func(bench_name, bench_func, None)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
