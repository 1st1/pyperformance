#
# This source file is part of the EdgeDB open source project.
#
# Copyright 2008-present MagicStack Inc. and the EdgeDB authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Pure-Python replacement for the ``edb._edgeql_parser`` Rust extension.

In Gel proper this module is a PyO3 extension built from the
``edb/edgeql-parser`` crate.  For the purposes of the pyperformance
benchmark it has been ported to Python, keeping the same module-level API
so that the rest of the (vendored) Gel code can be used unmodified:

* the tokenizer is a port of ``edgeql-parser/src/tokenizer.rs`` and
  ``validation.rs``;
* the parser is a port of the LR(1) driver in
  ``edgeql-parser/src/parser/mod.rs``, *without* the error recovery
  machinery: the first unexpected token is reported as a syntax error;
* the grammar spec is loaded from gzipped JSON (the output of
  ``edb.common.parsing.spec_to_json()``) instead of bincode.

Query normalization (constant extraction) and token (de)serialization
are only needed by the server and are not implemented.
"""

from __future__ import annotations
from typing import Any, Callable, Optional

import base64
import decimal
import gzip
import hashlib
import json
import re
import unicodedata


class SyntaxError(Exception):
    pass


# -------------------------------------------------------------------------
# Keywords (keywords.rs)
# -------------------------------------------------------------------------

unreserved_keywords = frozenset((
    "abort",
    "abstract",
    "access",
    "after",
    "alias",
    "allow",
    "all",
    "annotation",
    "applied",
    "as",
    "asc",
    "assignment",
    "before",
    "blobal",
    "branch",
    "cardinality",
    "cast",
    "committed",
    "config",
    "conflict",
    "constraint",
    "cube",
    "current",
    "data",
    "database",
    "ddl",
    "declare",
    "default",
    "deferrable",
    "deferred",
    "delegated",
    "desc",
    "deny",
    "each",
    "empty",
    "expression",
    "extension",
    "final",
    "first",
    "force",
    "from",
    "function",
    "future",
    "implicit",
    "index",
    "infix",
    "inheritable",
    "instance",
    "into",
    "isolation",
    "json",
    "last",
    "link",
    "migration",
    "multi",
    "named",
    "object",
    "of",
    "only",
    "onto",
    "operator",
    "optionality",
    "order",
    "orphan",
    "overloaded",
    "owned",
    "package",
    "permission",
    "policy",
    "populate",
    "postfix",
    "prefix",
    "property",
    "proposed",
    "pseudo",
    "read",
    "reject",
    "release",
    "rename",
    "repeatable",
    "required",
    "reset",
    "restrict",
    "rewrite",
    "role",
    "roles",
    "rollup",
    "savepoint",
    "scalar",
    "schema",
    "sdl",
    "serializable",
    "session",
    "source",
    "superuser",
    "system",
    "target",
    "template",
    "ternary",
    "text",
    "then",
    "to",
    "transaction",
    "trigger",
    "type",
    "unless",
    "using",
    "verbose",
    "version",
    "view",
    "write",
))

partial_reserved_keywords = frozenset(("except", "intersect", "union"))

future_reserved_keywords = frozenset((
    "anyarray",
    "begin",
    "case",
    "check",
    "deallocate",
    "discard",
    "end",
    "explain",
    "fetch",
    "get",
    "global",
    "grant",
    "import",
    "listen",
    "load",
    "lock",
    "match",
    "move",
    "notify",
    "on",
    "over",
    "prepare",
    "partition",
    "raise",
    "refresh",
    "revoke",
    "single",
    "when",
    "window",
    "never",
))

current_reserved_keywords = frozenset((
    "__source__",
    "__subject__",
    "__type__",
    "__std__",
    "__edgedbsys__",
    "__edgedbtpl__",
    "__new__",
    "__old__",
    "__specified__",
    "__default__",
    "administer",
    "alter",
    "analyze",
    "and",
    "anytuple",
    "anytype",
    "anyobject",
    "by",
    "commit",
    "configure",
    "create",
    "delete",
    "describe",
    "detached",
    "distinct",
    "do",
    "drop",
    "else",
    "exists",
    "extending",
    "false",
    "filter",
    "for",
    "group",
    "if",
    "ilike",
    "in",
    "insert",
    "introspect",
    "is",
    "like",
    "limit",
    "module",
    "not",
    "offset",
    "optional",
    "or",
    "rollback",
    "select",
    "set",
    "start",
    "true",
    "typeof",
    "update",
    "variadic",
    "with",
))

_combined_keywords = frozenset((
    "named only",
    "set annotation",
    "set type",
    "extension package",
    "order by",
))

_all_keywords = (
    unreserved_keywords
    | partial_reserved_keywords
    | future_reserved_keywords
    | current_reserved_keywords
    | _combined_keywords
)

_reserved_keywords = future_reserved_keywords | current_reserved_keywords

# Current max keyword length is 10, but we're reserving some space
_MAX_KEYWORD_LENGTH = 16


# -------------------------------------------------------------------------
# Tokens
# -------------------------------------------------------------------------
#
# Token kinds are represented as interned strings:
#
# * punctuation and operators use their literal text ("+", ":=", etc);
# * keywords use their lower-cased text prefixed with "kw:";
# * everything else uses the token names used by the Python grammar
#   ("IDENT", "SCONST", "EOI", etc).

K_IDENT = "IDENT"
K_EOI = "EOI"
K_ICONST = "ICONST"
K_FCONST = "FCONST"
K_NICONST = "NICONST"
K_NFCONST = "NFCONST"
K_BCONST = "BCONST"
K_SCONST = "SCONST"
K_STRINTERPSTART = "STRINTERPSTART"
K_STRINTERPCONT = "STRINTERPCONT"
K_STRINTERPEND = "STRINTERPEND"
K_PARAMETER = "PARAMETER"
K_PARAMETERANDTYPE = "PARAMETERANDTYPE"
K_SUBSTITUTION = "SUBSTITUTION"
K_BACKTICKNAME = "BACKTICKNAME"

_KW_KINDS = {kw: "kw:" + kw for kw in _all_keywords}

_NAMED_KINDS = frozenset((
    K_IDENT, K_EOI, K_ICONST, K_FCONST, K_NICONST, K_NFCONST, K_BCONST,
    K_SCONST, K_STRINTERPSTART, K_STRINTERPCONT, K_STRINTERPEND,
    K_PARAMETER, K_PARAMETERANDTYPE, K_SUBSTITUTION,
    "STARTBLOCK", "STARTEXTENSION", "STARTFRAGMENT", "STARTMIGRATION",
    "STARTSDLDOCUMENT",
))

_USER_FRIENDLY = {
    K_IDENT: "identifier",
    K_EOI: "end of input",
    K_BCONST: "binary constant",
    K_FCONST: "float constant",
    K_ICONST: "int constant",
    K_NFCONST: "decimal constant",
    K_NICONST: "big int constant",
    K_SCONST: "string constant",
}


class OpaqueToken:
    """A token produced by the tokenizer (and a CST terminal)."""

    __slots__ = ('kind', 'text', 'value', 'start', 'end')

    def __init__(
        self, kind: str, text: str, value: Any, start: int, end: int
    ) -> None:
        self.kind = kind
        self.text = text
        self.value = value
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f'{self.text}[{self.kind}]'

    def span_start(self) -> int:
        return self.start

    def span_end(self) -> int:
        return self.end

    def is_ident(self) -> bool:
        return self.kind == K_IDENT


Terminal = OpaqueToken


class _TokenizerError(Exception):
    def __init__(
        self,
        message: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.start = start
        self.end = end


_RUST_ESCAPES = {
    '\0': '\\0',
    '\t': '\\t',
    '\r': '\\r',
    '\n': '\\n',
    '\\': '\\\\',
}


def _rust_escape_debug(s: str, quote: str) -> str:
    # An approximation of Rust's `{:?}` for strings and chars.
    res = []
    for c in s:
        esc = _RUST_ESCAPES.get(c)
        if esc is not None:
            res.append(esc)
        elif c == quote:
            res.append('\\' + c)
        elif not c.isprintable():
            res.append(f'\\u{{{ord(c):x}}}')
        else:
            res.append(c)
    return quote + ''.join(res) + quote


def _rust_repr(s: str) -> str:
    return _rust_escape_debug(s, '"')


def _rust_char_repr(c: str) -> str:
    return _rust_escape_debug(c, "'")


_PROHIBITED_RE = re.compile('[\0\u202A-\u202E\u2066-\u2069]')

_WS_RE = re.compile('[ \t\r\n\ufeff]+')
_COMMENT_RE = re.compile(r'#[^\r\n]*[\r\n]?')

_OPERATOR_RE = re.compile(
    r'''
        \.\?> | \?!=
      | := | :: | -> | -= | >= | <= | \+= | \+\+ | // | \.< | \?\? | \?=
      | != | \*\*
      | [:\-><+/.=,()\[\]{};*%^&|@]
    ''',
    re.X,
)

_OPERATOR_CHARS = frozenset(':-><+/.=,()[]{};*%^&|@')

_IDENT_RE = re.compile(r'[^\W\d]\w*')
_BACKTICK_RE = re.compile(r'`(?:[^`]|``)*`')
_SUBST_RE = re.compile(r'\\\((\w*)(.?)', re.S)
_DIGITS_RE = re.compile(r'[0-9]+')
_INT_PART_RE = re.compile(r'[0-9][0-9_]*')
_FRAC_PART_RE = re.compile(r'\.([0-9_]*)')
_EXP_PART_RE = re.compile(r'e([+-]?)([0-9]?)[0-9_]*')
_WORD_RE = re.compile(r'\w+')
_PARAM_RE = re.compile(r'\$\w+')

_STR_RES = {
    # (quote, raw)
    ("'", False): re.compile(r"'(?:[^'\\]|\\[^(])*(?:'|\\\()", re.S),
    ('"', False): re.compile(r'"(?:[^"\\]|\\[^(])*(?:"|\\\()', re.S),
    ("'", True): re.compile(r"'[^']*'", re.S),
    ('"', True): re.compile(r'"[^"]*"', re.S),
}

_BIN_STR_RES = {
    ("'", False): re.compile(r"'(?:[^'\\]|\\.)*'", re.S),
    ('"', False): re.compile(r'"(?:[^"\\]|\\.)*"', re.S),
    ("'", True): re.compile(r"'[^']*'", re.S),
    ('"', True): re.compile(r'"[^"]*"', re.S),
}

_STR_INTERP_CONT_RES = {
    "'": re.compile(r"\)(?:[^'\\]|\\[^(])*(?:'|\\\()", re.S),
    '"': re.compile(r'\)(?:[^"\\]|\\[^(])*(?:"|\\\()', re.S),
}


def _check_prohibited(text: str, escape: bool) -> None:
    m = _PROHIBITED_RE.search(text)
    if m is None:
        return
    c = ord(m.group(0))
    if c == 0 and escape:
        raise _TokenizerError("character U+0000 is not allowed")
    elif escape:
        raise _TokenizerError(
            f"character U+{c:04X} is not allowed, "
            f"use escaped form \\u{c:04x}"
        )
    else:
        raise _TokenizerError(f"character U+{c:04X} is not allowed")


class _Tokenizer:
    """Port of ``Tokenizer`` from tokenizer.rs.

    Yields ``(kind, text, start, end)`` tuples, where ``start`` and
    ``end`` are *character* offsets into the source string.
    """

    def __init__(self, buf: str) -> None:
        self.buf = buf
        self.off = 0
        self.dot = False
        # We maintain a stack of the starting string characters and
        # parentheses nesting level for all our open string
        # interpolations, since we need to match the correct one when
        # closing them.
        self.str_interp_stack: list[tuple[str, int]] = []
        # The number of currently open parentheses. If we see a close
        # paren when there are no open parens *and* we are inside a
        # string inerpolation, we close it.
        self.open_parens = 0
        self.skip_whitespace()

    def __iter__(self):
        return self

    def __next__(self) -> tuple[str, str, int, int]:
        buf = self.buf
        start = self.off
        if start >= len(buf):
            raise StopIteration

        try:
            kind, length = self.peek_token()
        except _TokenizerError as e:
            if e.start is None:
                e.start = start
                e.end = start
            raise

        if kind == '(':
            self.open_parens += 1
        elif kind == ')':
            if self.open_parens > 0:
                self.open_parens -= 1
        elif kind is K_STRINTERPSTART:
            self.str_interp_stack.append((buf[start], self.open_parens))
        elif kind is K_STRINTERPEND:
            self.str_interp_stack.pop()

        end = start + length
        self.off = end
        self.dot = kind == '.'
        self.skip_whitespace()
        return kind, buf[start:end], start, end

    def skip_whitespace(self) -> None:
        buf = self.buf
        off = self.off
        n = len(buf)
        while off < n:
            c = buf[off]
            if c == '#':
                m = _COMMENT_RE.match(buf, off)
                bad = _PROHIBITED_RE.search(m.group(0), 1)
                if bad is not None:
                    # can't return error from skip_whitespace
                    # but we return up to this char, so the tokenizer
                    # chokes on it next time is invoked
                    off += bad.start()
                    break
                off = m.end()
            elif c in ' \t\r\n\ufeff':
                off = _WS_RE.match(buf, off).end()
            else:
                break
        self.off = off

    def peek_token(self) -> tuple[str, int]:
        buf = self.buf
        off = self.off
        c = buf[off]

        if c in _OPERATOR_CHARS:
            if c == ')' and self.str_interp_stack:
                delim, paren_count = self.str_interp_stack[-1]
                if paren_count == self.open_parens:
                    return self.parse_string_interp_cont(delim)
            m = _OPERATOR_RE.match(buf, off)
            op = m.group(0)
            if op == '.' and buf.startswith('.?', off):
                raise _TokenizerError(
                    "`.?` is not an operator, did you mean `.?>` ?")
            return op, len(op)

        if c == '_' or c.isalpha():
            m = _IDENT_RE.match(buf, off)
            end = m.end()
            val = m.group(0)
            if end < len(buf):
                nc = buf[end]
                if nc == '"' or nc == "'":
                    if val == 'r':
                        return self.parse_string(end - off, True, False)
                    elif val == 'b':
                        return self.parse_string(end - off, False, True)
                    elif val == 'rb' or val == 'br':
                        return self.parse_string(end - off, True, True)
                    else:
                        raise _TokenizerError(
                            f"prefix {_rust_repr(val)} is not allowed for "
                            f"strings, allowed: `b`, `r`"
                        )
                elif nc == '`':
                    raise _TokenizerError(
                        f"prefix {_rust_repr(val)} is not allowed for "
                        f"field names, perhaps missing comma or dot?"
                    )
            if len(val) <= _MAX_KEYWORD_LENGTH:
                kw = _KW_KINDS.get(val.lower())
                if kw is not None and val.isascii():
                    return kw, end - off
            if val.startswith('__') and val.endswith('__'):
                raise _TokenizerError(
                    "identifiers surrounded by double underscores "
                    "are forbidden"
                )
            return K_IDENT, end - off

        if c == '"' or c == "'":
            return self.parse_string(0, False, False)

        if '0' <= c <= '9':
            if self.dot:
                m = _DIGITS_RE.match(buf, off)
                end = m.end()
                if end < len(buf) and buf[end].isalpha():
                    raise _TokenizerError(
                        f"unexpected char {_rust_char_repr(buf[end])}, "
                        f"only integers are allowed after dot "
                        f"(for tuple access)"
                    )
                if c == '0' and end - off > 1:
                    raise _TokenizerError(
                        "leading zeros are not allowed in numbers")
                return K_ICONST, end - off
            else:
                return self.parse_number()

        if c == '?':
            nc = buf[off + 1:off + 2]
            if nc == '?':
                return '??', 2
            elif nc == '=':
                return '?=', 2
            elif nc == '!':
                if buf[off + 2:off + 3] == '=':
                    return '?!=', 3
                raise _TokenizerError(
                    "`?!` is not an operator, did you mean `?!=` ?")
            raise _TokenizerError(
                "Bare `?` is not an operator, did you mean `?=` or `??` ?")

        if c == '!':
            if buf[off + 1:off + 2] == '=':
                return '!=', 2
            raise _TokenizerError(
                "Bare `!` is not an operator, did you mean `!=`?")

        if c == '`':
            m = _BACKTICK_RE.match(buf, off)
            if m is None:
                _check_prohibited(buf[off:], False)
                raise _TokenizerError("unterminated backtick name")
            val = m.group(0)
            _check_prohibited(val, False)
            if val.startswith("`@"):
                raise _TokenizerError(
                    "backtick-quoted name cannot start with char `@`")
            if val.startswith("`$"):
                raise _TokenizerError(
                    "backtick-quoted name cannot start with char `$`")
            if "::" in val:
                raise _TokenizerError(
                    "backtick-quoted name cannot contain `::`")
            if val.startswith("`__") and val.endswith("__`"):
                raise _TokenizerError(
                    "backtick-quoted names surrounded by double "
                    "underscores are forbidden"
                )
            if len(val) == 2:
                raise _TokenizerError("backtick quotes cannot be empty")
            return K_BACKTICKNAME, len(val)

        if c == '$':
            return self.parse_dollar()

        if c == '\\':
            m = _SUBST_RE.match(buf, off)
            if m is None:
                raise _TokenizerError(
                    f"unexpected character {_rust_char_repr(c)}")
            term = m.group(2)
            if term == ')':
                return K_SUBSTITUTION, m.end() - off
            elif term == '':
                raise _TokenizerError("unclosed \\(name) token")
            else:
                raise _TokenizerError(
                    "only alphanumerics are allowed in \\(name) token")

        raise _TokenizerError(f"unexpected character {_rust_char_repr(c)}")

    def parse_dollar(self) -> tuple[str, int]:
        buf = self.buf
        off = self.off
        nc = buf[off + 1:off + 2]

        if nc == '$':
            end = buf.find('$$', off + 2)
            if end < 0:
                raise _TokenizerError("unterminated string started with $$")
            _check_prohibited(buf[off + 2:end], False)
            return K_SCONST, end + 2 - off

        if nc == '`':
            m = _BACKTICK_RE.match(buf, off + 1)
            if m is None:
                _check_prohibited(buf[off + 1:], False)
                raise _TokenizerError("unterminated backtick argument")
            var = buf[off:m.end()]
            _check_prohibited(var, False)
            if var.startswith("$`@"):
                raise _TokenizerError(
                    "backtick-quoted argument cannot start with char `@`")
            if "::" in var:
                raise _TokenizerError(
                    "backtick-quoted argument cannot contain `::`")
            if var.startswith("$`__") and var.endswith("__`"):
                raise _TokenizerError(
                    "backtick-quoted arguments surrounded by double "
                    "underscores are forbidden"
                )
            if len(var) == 3:
                raise _TokenizerError(
                    "backtick-quoted argument cannot be empty")
            return K_PARAMETER, len(var)

        m = _PARAM_RE.match(buf, off)
        if m is None:
            raise _TokenizerError("bare $ is not allowed")

        end = m.end()
        name = buf[off + 1:end]
        if end < len(buf) and buf[end] == '$':
            # dollar-quoted string
            marker = buf[off:end + 1]
            msize = len(marker)
            if '0' <= name[0] <= '9':
                raise _TokenizerError(
                    "dollar quote must not start with a digit")
            if not marker.isascii():
                raise _TokenizerError(
                    "dollar quote supports only ascii chars")
            close = buf.find(marker, off + msize)
            if close < 0:
                raise _TokenizerError(
                    f"unterminated string started with {_rust_repr(marker)}"
                )
            _check_prohibited(buf[off + msize:close], False)
            return K_SCONST, close + msize - off

        if not name.isdecimal() and '0' <= name[0] <= '9':
            raise _TokenizerError(
                f"the {_rust_repr(buf[off:end])} is not a valid argument, "
                f"either name starting with letter or only digits "
                f"are expected"
            )
        return K_PARAMETER, end - off

    def parse_string(
        self, quote_off: int, raw: bool, binary: bool
    ) -> tuple[str, int]:
        buf = self.buf
        start = self.off + quote_off
        open_quote = buf[start]
        if binary:
            m = _BIN_STR_RES[open_quote, raw].match(buf, start)
            text = m.group(0) if m is not None else buf[start:]
            if not text.isascii():
                for c in text:
                    if ord(c) > 0x7f:
                        raise _TokenizerError(
                            f"invalid bytes literal: character "
                            f"{_rust_char_repr(c)} is unexpected, "
                            f"only ascii chars are allowed in bytes literals"
                        )
            if m is not None:
                return K_BCONST, m.end() - self.off
        else:
            m = _STR_RES[open_quote, raw].match(buf, start)
            if m is not None:
                text = m.group(0)
                _check_prohibited(text, True)
                if not raw and text.endswith('\\('):
                    return K_STRINTERPSTART, m.end() - self.off
                return K_SCONST, m.end() - self.off
            _check_prohibited(buf[start:], True)
        raise _TokenizerError(
            f"unterminated string, quoted by `{open_quote}`")

    def parse_string_interp_cont(self, end: str) -> tuple[str, int]:
        buf = self.buf
        m = _STR_INTERP_CONT_RES[end].match(buf, self.off)
        if m is not None:
            text = m.group(0)
            _check_prohibited(text, True)
            if text.endswith('\\('):
                return K_STRINTERPCONT, len(text)
            return K_STRINTERPEND, len(text)
        _check_prohibited(buf[self.off:], True)
        raise _TokenizerError(
            f"unterminated string with interpolations, quoted by `{end}`")

    def parse_number(self) -> tuple[str, int]:
        buf = self.buf
        off = self.off
        n = len(buf)

        pos = _INT_PART_RE.match(buf, off).end()
        nc = buf[pos] if pos < n else ''
        int_len = pos - off
        if buf[off] == '0' and int_len > 1:
            raise _TokenizerError(
                "unexpected leading zeros are not allowed in numbers")

        decimal = False
        is_float = False

        if nc == '.':
            is_float = True
            decimal = True
            m = _FRAC_PART_RE.match(buf, pos)
            frac = m.group(1)
            pos = m.end()
            nc = buf[pos] if pos < n else ''
            if frac.startswith('_'):
                raise _TokenizerError(
                    "expected digit after dot, found underscore")
            if nc == '.':
                raise _TokenizerError(
                    "unexpected extra decimal dot in number")
            if not frac:
                if nc == 'e':
                    raise _TokenizerError(
                        "expected digit after dot, found exponent")
                elif nc.isalpha():
                    raise _TokenizerError(
                        "expected digit after dot, found suffix")
                else:
                    raise _TokenizerError(
                        "expected digit after dot, found end of decimal")

        if nc == 'e':
            is_float = True
            m = _EXP_PART_RE.match(buf, pos)
            sign, digit = m.groups()
            if not digit:
                after = buf[pos + 1 + len(sign):pos + 2 + len(sign)]
                if sign and after == '.':
                    raise _TokenizerError(
                        "unexpected extra decimal dot in number")
                raise _TokenizerError(
                    "unexpected optional `+` or `-` followed by digits "
                    "must follow `e` in float const"
                )
            if sign == '-':
                decimal = True
            pos = m.end()
            nc = buf[pos] if pos < n else ''
            if nc == '.':
                raise _TokenizerError(
                    "unexpected extra decimal dot in number")

        if not nc.isalpha():
            if is_float:
                return K_FCONST, pos - off
            else:
                return K_ICONST, pos - off

        soff = pos - off
        end = _WORD_RE.match(buf, pos).end()
        suffix = buf[pos:end]
        if suffix == 'n':
            if decimal:
                return K_NFCONST, end - off
            else:
                return K_NICONST, end - off

        if len(suffix) > 8:
            suffix = suffix[:8] + '...'
        val = buf[off:pos] if soff < 20 else '123'
        if suffix.startswith('O'):
            raise _TokenizerError(
                f"suffix {_rust_repr(suffix)} is invalid for numbers, "
                f"perhaps mixed up letter `O` with zero `0`?"
            )
        elif decimal:
            raise _TokenizerError(
                f"suffix {_rust_repr(suffix)} is invalid for numbers, "
                f"perhaps you wanted `{val}n` (decimal)?"
            )
        else:
            raise _TokenizerError(
                f"suffix {_rust_repr(suffix)} is invalid for numbers, "
                f"perhaps you wanted `{val}n` (bigint)?"
            )


# -------------------------------------------------------------------------
# Value parsing (validation.rs, helpers/strings.rs, helpers/bytes.rs)
# -------------------------------------------------------------------------

_SIMPLE_ESCAPES = {
    '"': '"',
    '\\': '\\',
    '/': '/',
    "'": "'",
    'b': '\b',
    'f': '\f',
    'n': '\n',
    'r': '\r',
    't': '\t',
}

_HEX2_RE = re.compile(r'[0-9a-fA-F]{2}')
_HEX4_RE = re.compile(r'[0-9a-fA-F]{4}')
_HEX8_RE = re.compile(r'[0-9a-fA-F]{8}')


def _escape_debug(s: str) -> str:
    return _rust_escape_debug(s, '')


def _unquote_string_inner(s: str) -> str:
    if '\\' not in s:
        return s

    res = []
    n = len(s)
    i = 0
    while i < n:
        j = s.find('\\', i)
        if j < 0:
            res.append(s[i:])
            break
        if j > i:
            res.append(s[i:j])
        if j + 1 >= n:
            raise ValueError("quoted string cannot end in slash")
        c = s[j + 1]
        i = j + 2
        simple = _SIMPLE_ESCAPES.get(c)
        if simple is not None:
            res.append(simple)
        elif c == 'x':
            m = _HEX2_RE.match(s, i)
            if m is None:
                raise ValueError(
                    f"invalid string literal: invalid escape sequence "
                    f"'\\x{_escape_debug(s[i:i + 2])}'"
                )
            code = int(m.group(0), 16)
            if code > 0x7f or code == 0:
                raise ValueError(
                    f"invalid string literal: invalid escape sequence "
                    f"'\\x{code:x}' (only non-null ascii allowed)"
                )
            res.append(chr(code))
            i += 2
        elif c == 'u' or c == 'U':
            size = 4 if c == 'u' else 8
            m = (_HEX4_RE if c == 'u' else _HEX8_RE).match(s, i)
            code = int(m.group(0), 16) if m is not None else 0
            if (
                m is None
                or code == 0
                or code > 0x10FFFF
                or 0xD800 <= code <= 0xDFFF
            ):
                raise ValueError(
                    f"invalid string literal: invalid escape sequence "
                    f"'\\{c}{_escape_debug(s[i:i + size])}'"
                )
            res.append(chr(code))
            i += size
        elif c == '\r' or c == '\n':
            # line continuation: skip all the following whitespace
            while i < n and s[i].isspace():
                i += 1
        else:
            raise ValueError(
                f"invalid string literal: invalid escape sequence "
                f"'\\{_escape_debug(c)}'"
            )

    return ''.join(res)


def unquote_string(value: str) -> str:
    if value.startswith('r'):
        return value[2:-1]
    elif value.startswith('$'):
        msize = value.find('$', 1)
        if msize < 0:
            raise ValueError("invalid dollar-quoted string")
        msize += 1
        return value[msize:len(value) - msize]
    else:
        end_trim = 2 if value.endswith('\\(') else 1
        return _unquote_string_inner(value[1:len(value) - end_trim])


_BYTES_SIMPLE_ESCAPES = {
    ord(k): ord(v) for k, v in _SIMPLE_ESCAPES.items()
}

_BYTES_WS = frozenset(b' \t\n\r\x0c')


def _unquote_bytes_inner(s: str) -> bytes:
    data = s.encode('utf-8')
    if b'\\' not in data:
        return data

    res = bytearray()
    n = len(data)
    i = 0
    while i < n:
        j = data.find(b'\\', i)
        if j < 0:
            res += data[i:]
            break
        res += data[i:j]
        c = data[j + 1]
        i = j + 2
        simple = _BYTES_SIMPLE_ESCAPES.get(c)
        if simple is not None:
            res.append(simple)
        elif c == 0x78:  # x
            hex = data[i:i + 2].decode('utf-8', 'replace')
            if _HEX2_RE.fullmatch(hex) is None:
                raise ValueError(
                    f"invalid bytes literal: invalid escape sequence "
                    f"'\\x{_escape_debug(hex)}'"
                )
            res.append(int(hex, 16))
            i += 2
        elif c == 0x0d or c == 0x0a:
            while i < n and data[i] in _BYTES_WS:
                i += 1
        else:
            raise ValueError(
                f"invalid bytes literal: invalid escape sequence "
                f"'\\{_escape_debug(chr(c))}'"
            )

    return bytes(res)


def unquote_bytes(value: str) -> bytes:
    idx = -1
    for i, c in enumerate(value):
        if c == "'" or c == '"':
            idx = i
            break
    if idx < 0:
        raise ValueError("invalid bytes literal: missing quotes")
    prefix = value[:idx]
    if prefix == 'br' or prefix == 'rb':
        return value[3:-1].encode('utf-8')
    elif prefix == 'b':
        return _unquote_bytes_inner(value[2:-1])
    else:
        raise ValueError(
            f"prefix {_rust_repr(prefix)} is not allowed for bytes, "
            f"allowed: `b`, `rb`"
        )


_INF = float('inf')
_STR_KINDS = frozenset((
    K_SCONST, K_STRINTERPSTART, K_STRINTERPCONT, K_STRINTERPEND,
))


def _parse_value(kind: str, text: str) -> Any:
    """Port of ``parse_value`` from validation.rs.

    Raises ValueError on invalid values.
    """
    if kind is K_IDENT or kind[:3] == 'kw:':
        return text
    elif kind in _STR_KINDS:
        return unquote_string(text)
    elif kind is K_ICONST:
        # We read unsigned here, because unary minus will only
        # be identified on the parser stage. And there is a number
        # -9223372036854775808 which can't be represented in
        # i64 as absolute (positive) value.
        num = int(text.replace('_', ''))
        if num > 0xFFFF_FFFF_FFFF_FFFF:
            raise ValueError(
                "error reading int: number too large to fit in target type")
        if num > 0x7FFF_FFFF_FFFF_FFFF:
            # emulate the `u64 as i64` cast
            num -= 0x1_0000_0000_0000_0000
        return num
    elif kind is K_PARAMETER:
        if text[1:2] == '`':
            return text[2:-1].replace('``', '`')
        else:
            return text[1:]
    elif kind is K_BACKTICKNAME:
        return text[1:-1].replace('``', '`')
    elif kind is K_FCONST:
        try:
            fnum = float(text.replace('_', ''))
        except ValueError as e:
            raise ValueError(f"can't parse std::float64: {e}") from None
        if fnum == _INF:
            raise ValueError("number is out of range for std::float64")
        if fnum == 0.0:
            mantissa = re.split('[eE]', text, maxsplit=1)[0]
            if any(c != '0' and c != '.' for c in mantissa):
                raise ValueError("number is out of range for std::float64")
        return fnum
    elif kind is K_NFCONST:
        try:
            # The Rust extension parses this into a BigDecimal, and then
            # converts it to Python as `float(str(value))`.
            return float(text[:-1].replace('_', ''))
        except ValueError as e:
            raise ValueError(f"can't parse decimal: {e}") from None
    elif kind is K_NICONST:
        try:
            dec = decimal.Decimal(text[:-1].replace('_', ''))
        except decimal.InvalidOperation as e:
            raise ValueError(f"error reading bigint: {e}") from None
        # this conversion to decimal and back to int
        # fixes thing like `1e2n` which we support for bigints
        if dec != dec.to_integral_value():
            raise ValueError("number is not integer")
        return int(dec)
    elif kind is K_BCONST:
        return unquote_bytes(text)
    elif kind is K_SUBSTITUTION:
        return text[2:-1]
    else:
        return None


_COMBINED_KW_PREFIXES = {
    'named': ('only',),
    'set': ('annotation', 'type'),
    'extension': ('package',),
    'order': ('by',),
}


def _byte_offsets(text: str) -> Optional[list[int]]:
    """Return a char offset -> utf-8 byte offset table for *text*."""
    if text.isascii():
        return None
    table = [0] * (len(text) + 1)
    off = 0
    for i, c in enumerate(text):
        table[i] = off
        o = ord(c)
        if o < 0x80:
            off += 1
        elif o < 0x800:
            off += 2
        elif o < 0x10000:
            off += 3
        else:
            off += 4
    table[len(text)] = off
    return table


def _tokenize(
    text: str,
) -> tuple[list[OpaqueToken], Optional[_TokenizerError]]:
    """Tokenize, validate and combine multi-word keywords.

    This is ``Tokenizer::new(s).validated_values().with_eof()``.
    Token spans are utf-8 *byte* offsets, same as in the Rust version.
    """
    offsets = _byte_offsets(text)
    tokens: list[OpaqueToken] = []
    append = tokens.append
    tokenizer = _Tokenizer(text)
    error: Optional[_TokenizerError] = None
    combine: Optional[tuple[str, ...]] = None
    try:
        for kind, tok_text, start, end in tokenizer:
            if combine is not None:
                # check if the previous and this tokens form
                # a multi-word keyword
                if kind is K_IDENT or kind[:3] == 'kw:':
                    lower = tok_text.lower()
                    if lower in combine:
                        prev = tokens[-1]
                        prev.text = f'{prev.text.lower()} {lower}'
                        prev.kind = _KW_KINDS[prev.text]
                        combine = None
                        continue
                combine = None

            try:
                value = _parse_value(kind, tok_text)
            except ValueError as e:
                raise _TokenizerError(str(e), start, end) from None

            if kind is K_IDENT or kind[:3] == 'kw:':
                if len(tok_text) <= _MAX_KEYWORD_LENGTH:
                    combine = _COMBINED_KW_PREFIXES.get(tok_text.lower())
            elif kind is K_BACKTICKNAME:
                kind = K_IDENT

            append(OpaqueToken(kind, tok_text, value, start, end))
    except _TokenizerError as e:
        error = e

    if error is None:
        pos = len(text)
        append(OpaqueToken(K_EOI, "", None, pos, pos))

    if offsets is not None:
        for tok in tokens:
            tok.start = offsets[tok.start]
            tok.end = offsets[tok.end]
        if error is not None:
            error.start = offsets[error.start]
            error.end = offsets[error.end]

    return tokens, error


class ParserResult:
    __slots__ = ('out', 'errors')

    def __init__(self, out: Any, errors: list[Any]) -> None:
        self.out = out
        self.errors = errors

    def pack(self) -> bytes:
        # The Rust extension serializes the tokens with bincode here.  The
        # serialized form is used for shipping tokens between the server
        # and the compiler pool, as well as for computing the cache key of
        # the source; we only care about the latter.
        hasher = hashlib.blake2b(b'\x00')
        for token in self.out:
            hasher.update(token.kind.encode('utf-8'))
            hasher.update(b'\x00')
            hasher.update(token.text.encode('utf-8'))
            hasher.update(b'\x00')
        return b'\x00' + hasher.digest()


def tokenize(s: str) -> ParserResult:
    tokens, error = _tokenize(s)
    errors = []
    if error is not None:
        errors.append((error.message, (error.start, error.end), None, None))
    return ParserResult(tokens, errors)


class Entry:
    pass


def normalize(text: str) -> Entry:
    raise NotImplementedError(
        'query normalization is not implemented in the pure-Python '
        'EdgeQL parser'
    )


def unpack(serialized: bytes) -> Any:
    raise NotImplementedError(
        'token deserialization is not implemented in the pure-Python '
        'EdgeQL parser'
    )


def unpickle_token(data: bytes) -> OpaqueToken:
    raise NotImplementedError(
        'token deserialization is not implemented in the pure-Python '
        'EdgeQL parser'
    )


# -------------------------------------------------------------------------
# Parser (parser/mod.rs, parser/spec.rs, parser/cst.rs)
# -------------------------------------------------------------------------

class CSTNode:
    __slots__ = ('production', 'terminal')

    def __init__(
        self,
        production: Optional[Production],
        terminal: Optional[Terminal],
    ) -> None:
        self.production = production
        self.terminal = terminal


class Production:
    __slots__ = ('id', 'args', 'start', 'end')

    def __init__(
        self,
        id: int,
        args: list[CSTNode],
        start: Optional[int],
        end: Optional[int],
    ) -> None:
        self.id = id
        self.args = args
        self.start = start
        self.end = end


class _Spec:
    __slots__ = ('actions', 'goto', 'inlines', 'production_names')

    def __init__(self, data: dict[str, Any]) -> None:
        # An action is either an int (shift and go to that state)
        # or a (production_id, non_term, cnt, inline_position) tuple
        # (reduce).
        inlines = {int(k): int(v) for k, v in data['inlines']}
        reduces: dict[tuple[int, str, int], Any] = {}

        actions: list[dict[str, Any]] = []
        for st_actions in data['actions']:
            out = {}
            for token_name, action in st_actions:
                kind = _get_token_kind(token_name)
                shift = action.get('Shift')
                if shift is not None:
                    out[kind] = int(shift)
                else:
                    reduce = action['Reduce']
                    key = (
                        reduce['production_id'],
                        reduce['non_term'],
                        reduce['cnt'],
                    )
                    try:
                        out[kind] = reduces[key]
                    except KeyError:
                        out[kind] = reduces[key] = (
                            *key, inlines.get(key[0]),
                        )
            actions.append(out)

        self.actions = actions
        self.goto = [dict(st_goto) for st_goto in data['goto']]
        self.inlines = inlines
        self.production_names = [
            (cls_name, meth_name)
            for cls_name, meth_name in data['production_names']
        ]


def _get_token_kind(token_name: str) -> str:
    if token_name == '<$>':
        return K_EOI
    elif token_name in _NAMED_KINDS or token_name == '<e>':
        return token_name
    elif not token_name[0].isalpha():
        # punctuation
        return token_name
    else:
        kw = token_name.lower()
        if kw.startswith('dunder'):
            kw = f'__{kw[len("dunder"):]}__'
        return _KW_KINDS[kw]


_SPEC: Optional[tuple[_Spec, list[tuple[type, Callable]]]] = None


def preload_spec(spec_filepath: str) -> None:
    """Loads the grammar specification from file and caches it in memory."""
    global _SPEC

    if _SPEC is not None:
        return

    with gzip.open(spec_filepath, 'rb') as f:
        spec = _Spec(json.load(f))

    import importlib
    grammar_mod = importlib.import_module('edb.edgeql.parser.grammar.start')
    from edb.common import parsing
    productions = parsing.load_spec_productions(
        spec.production_names, grammar_mod)

    _SPEC = (spec, productions)


def save_spec(spec_json: str, dst: str) -> None:
    """Serialize the grammar specification and write it to a file."""
    # validate
    _Spec(json.loads(spec_json))
    # mtime=0 makes the output reproducible
    with open(dst, 'wb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as gz:
            gz.write(spec_json.encode('utf-8'))


def _get_spec() -> tuple[_Spec, list[tuple[type, Callable]]]:
    if _SPEC is None:
        raise AssertionError("grammar spec not loaded")
    return _SPEC


def _quote_name(s: str) -> str:
    if s.replace('_', 'a').isalnum():
        lower = s.lower()
        if (
            lower not in _reserved_keywords
            and lower not in partial_reserved_keywords
        ):
            return s
    escaped = s.replace('`', '``')
    return f'`{escaped}`'


def _describe_terminal(token: Terminal) -> str:
    kind = token.kind
    if not token.text:
        friendly = _USER_FRIENDLY.get(kind)
        if friendly is not None:
            return friendly
    if kind is K_IDENT:
        return f"'{_quote_name(token.text)}'"
    elif kind[:3] == 'kw:':
        return f"keyword '{kind[3:].upper()}'"
    else:
        return f"'{token.text}'"


def parse(
    start_token_name: str,
    tokens: list[OpaqueToken],
) -> tuple[ParserResult, list[tuple[type, Callable]]]:
    spec, productions = _get_spec()
    actions = spec.actions
    goto = spec.goto

    start = Terminal(_get_token_kind(start_token_name), "", None, 0, 0)

    # Append EIO token.
    # We have a weird setup that requires two EOI tokens:
    # - one is consumed by the grammar generator and does not contribute to
    #   span of the nodes.
    # - second is consumed by explicit EOF tokens in EdgeQLGrammar NonTerm.
    #   Since these are children of productions, they do contribute to the
    #   spans of the top-level nodes.
    # First EOI is produced by tokenizer (with correct offset) and second one
    # is injected here.
    end = tokens[-1].end if tokens else 0
    eoi = Terminal(K_EOI, "", None, end, end)

    # The parser stack.
    states = [0]
    nodes: list[CSTNode] = []

    for token_list in ((start,), tokens, (eoi,)):
        for token in token_list:
            kind = token.kind
            while True:
                # find next action
                action = actions[states[-1]].get(kind)
                if action.__class__ is int:
                    # shift
                    states.append(action)
                    nodes.append(CSTNode(None, token))
                    break
                elif action is None:
                    # Unlike the original, there is no error recovery here.
                    error = (
                        f'Unexpected {_describe_terminal(token)}',
                        (token.start, token.end),
                        None,
                        None,
                    )
                    return ParserResult(None, [error]), productions

                # reduce
                production_id, non_term, cnt, inline_position = action

                if cnt:
                    args = nodes[-cnt:]
                    del nodes[-cnt:]
                    del states[-cnt:]
                else:
                    args = []

                if inline_position is not None:
                    # inline (if there is an inlining rule)
                    node = args[inline_position]
                else:
                    # the span of syntactically ordered nodes
                    span_start = span_end = None
                    for arg in args:
                        terminal = arg.terminal
                        if terminal is not None:
                            span_start = terminal.start
                            break
                        span_start = arg.production.start
                        if span_start is not None:
                            break
                    if span_start is not None:
                        for arg in reversed(args):
                            terminal = arg.terminal
                            if terminal is not None:
                                span_end = terminal.end
                                break
                            span_end = arg.production.end
                            if span_end is not None:
                                break
                    node = CSTNode(
                        Production(production_id, args, span_start, span_end),
                        None,
                    )

                states.append(goto[states[-1]][non_term])
                nodes.append(node)

    # pop the EOI from the top of the stack
    final_eoi = nodes.pop()
    assert (
        final_eoi.terminal is not None and final_eoi.terminal.kind is K_EOI
    ), "expected EOI CST node"
    assert len(nodes) == 1, "expected a single CST node"

    return ParserResult(nodes[0], []), productions


def suggest_next_keywords(
    start_token_name: str, tokens: list[OpaqueToken]
) -> tuple[list[str], bool]:
    raise NotImplementedError(
        'keyword suggestions are not implemented in the pure-Python '
        'EdgeQL parser'
    )


# -------------------------------------------------------------------------
# Source positions (position.rs)
# -------------------------------------------------------------------------

def _char_width(c: str) -> int:
    if unicodedata.combining(c):
        return 0
    if unicodedata.east_asian_width(c) in ('W', 'F'):
        return 2
    return 1


def _new_lines_in_fragment(data: bytes) -> int:
    was_lf = False
    lines = 0
    for byte in data:
        if byte == 0x0a:
            if was_lf:
                was_lf = False
            else:
                lines += 1
        elif byte == 0x0d:
            lines += 1
            was_lf = True
        else:
            was_lf = False
    return lines


class SourcePoint:
    __slots__ = (
        'zero_based_line', 'zero_based_column', 'utf16column', 'offset',
        'char_offset',
    )

    def __init__(
        self,
        line: int,
        column: int,
        utf16column: int,
        offset: int,
        char_offset: int,
    ) -> None:
        self.zero_based_line = line
        self.zero_based_column = column
        self.utf16column = utf16column
        self.offset = offset
        self.char_offset = char_offset

    @property
    def line(self) -> int:
        return self.zero_based_line + 1

    @property
    def column(self) -> int:
        return self.zero_based_column + 1

    @staticmethod
    def from_offsets(data: bytes, offsets: list[int]) -> list[SourcePoint]:
        result = []
        for offset in sorted(offsets):
            if offset > len(data):
                raise RuntimeError("offset out of range")
            prefix = data[:offset]
            try:
                prefix_s = prefix.decode('utf-8')
            except UnicodeDecodeError as e:
                raise RuntimeError(str(e)) from None
            loff = max(prefix.rfind(b'\r'), prefix.rfind(b'\n'))
            if loff >= 0:
                line_offset = loff + 1
                lines = prefix[:loff]
                if (
                    data[loff] == 0x0a and loff > 0 and data[loff - 1] == 0x0d
                ):
                    lines = lines[:-1]
                line = _new_lines_in_fragment(lines) + 1
            else:
                line = 0
                line_offset = 0
            col_s = prefix[line_offset:offset].decode('utf-8')
            if col_s.isascii():
                column = utf16column = len(col_s)
            else:
                column = sum(_char_width(c) for c in col_s)
                utf16column = len(col_s.encode('utf-16-le')) // 2
            result.append(
                SourcePoint(
                    line=line,
                    column=column,
                    utf16column=utf16column,
                    offset=offset,
                    char_offset=len(prefix_s),
                )
            )
        return result

    @staticmethod
    def from_lines_cols(
        data: bytes, lines_cols: list[tuple[int, int]]
    ) -> list[SourcePoint]:
        raise NotImplementedError(
            'SourcePoint.from_lines_cols() is not implemented in the '
            'pure-Python EdgeQL parser'
        )


def offset_of_line(text: str, target: int) -> int:
    was_lf = False
    line = 0  # this assumes line found by rfind
    data = text.encode('utf-8')
    for idx, byte in enumerate(data):
        if line >= target:
            return idx
        if byte == 0x0a:
            line += 1
            was_lf = False
        elif was_lf:
            line += 1
            if line >= target:
                return idx
            was_lf = byte == 0x0d
        elif byte == 0x0d:
            was_lf = True
    if was_lf:
        line += 1
    if target > line:
        raise IndexError("line number is too large")
    return len(data)


# -------------------------------------------------------------------------
# Migration hashing (hash.rs)
# -------------------------------------------------------------------------

class Hasher:
    def __init__(self, hasher: Any) -> None:
        self._hasher = hasher

    @staticmethod
    def start_migration(parent_id: str) -> Hasher:
        hasher = hashlib.sha256()
        hasher.update(b"CREATE\0MIGRATION\0ONTO\0")
        hasher.update(parent_id.encode('utf-8'))
        hasher.update(b"\0{\0")
        return Hasher(hasher)

    def add_source(self, data: str) -> None:
        if self._hasher is None:
            raise RuntimeError("cannot add source after finish")
        tokenizer = _Tokenizer(data)
        try:
            for _, text, _, _ in tokenizer:
                self._hasher.update(text.encode('utf-8'))
                self._hasher.update(b"\0")
        except _TokenizerError as e:
            offset = len(data[:tokenizer.off].encode('utf-8'))
            raise SyntaxError(e.message, (offset, None), None, None) from None

    def make_migration_id(self) -> str:
        if self._hasher is None:
            raise RuntimeError("cannot do migration id twice")
        hasher = self._hasher
        self._hasher = None
        hasher.update(b"}\0")
        hash = base64.b32encode(hasher.digest()).decode('ascii').rstrip('=')
        return f"m1{hash.lower()}"
