#!/usr/bin/env python3
"""Regenerate edb/edgeql/grammar.json.gz, the EdgeQL LR(1) parser tables.

This needs to be done every time the EdgeQL grammar definition in
edb/edgeql/parser/grammar/ is changed.  In Gel proper this is done
by `setup.py build_parsers`.
"""

import os.path
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edb import _edgeql_parser  # noqa: E402
from edb.common import parsing  # noqa: E402
from edb.edgeql import parser as qlparser  # noqa: E402
from edb.edgeql.parser import grammar as qlgrammar  # noqa: E402


def main():
    spec = parsing.load_parser_spec(qlgrammar.start)
    # The parser generator leaves a (useless) log file behind.
    log = os.path.splitext(qlgrammar.start.__file__)[0] + ".log"
    if os.path.exists(log):
        os.unlink(log)
    spec_json = parsing.spec_to_json(spec)
    _edgeql_parser.save_spec(spec_json, qlparser.get_spec_filepath())


if __name__ == "__main__":
    main()
