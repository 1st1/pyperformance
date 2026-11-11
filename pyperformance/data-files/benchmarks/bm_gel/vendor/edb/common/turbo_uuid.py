#
# This source file is part of the EdgeDB open source project.
#
# Copyright 2019-present MagicStack Inc. and the EdgeDB authors.
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

"""Pure-Python replacement for the ``edb.common.turbo_uuid`` extension.

In Gel proper this is a Cython extension (``edb/server/pgproto/uuid.pyx``)
implementing a faster ``uuid.UUID`` that can only be constructed from
``bytes`` or ``str``.
"""

from __future__ import annotations

import uuid


_setattr = object.__setattr__
_from_bytes = int.from_bytes
_unknown = uuid.SafeUUID.unknown


class UUID(uuid.UUID):

    __slots__ = ()

    def __init__(self, inp: bytes | str) -> None:
        if isinstance(inp, bytes):
            if len(inp) != 16:
                raise ValueError(f'16 bytes were expected, got {len(inp)}')
            _setattr(self, 'int', _from_bytes(inp, 'big'))
        elif isinstance(inp, str):
            hex = inp.replace('-', '')
            if len(hex) > 32:
                raise ValueError(
                    f'invalid UUID {inp!r}: decodes to more than 16 bytes')
            elif len(hex) < 32:
                raise ValueError(
                    f'invalid UUID {inp!r}: decodes to less than 16 bytes')
            try:
                _setattr(self, 'int', int(hex, 16))
            except ValueError:
                raise ValueError(f'invalid UUID {inp!r}') from None
        else:
            raise TypeError(f'a bytes or str object expected, got {inp!r}')
        _setattr(self, 'is_safe', _unknown)

    def __reduce__(self):
        return (type(self), (self.bytes,))
