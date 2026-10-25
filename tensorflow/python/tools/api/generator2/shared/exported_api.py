# [MODULE DOC]
# Data model and file format for "exported API" metadata used by the API generator. Defines
# immutable records for an exported symbol (source file, line, symbol name, v1 and v2 public
# names) and for an exported module docstring (source file, line, modules, text), plus a container
# class that holds sets of both and reads/writes them as a JSON file with a docs list and a
# symbols list. Shared between the extractor side that writes mapping files and the generator that
# reads them. No import-time side effects.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Reads and writes files with TF Python exports metadata."""

from collections.abc import Iterable, Sequence
import json
from typing import Any, NamedTuple


class ExportedSymbol(NamedTuple):
  # [CLASS DOC]
  # Immutable record (named tuple) describing one export-decorator use: the source file name, the
  # line number, the symbol name, and tuples of public API names for version 1 and version 2.
  # Hashable and orderable (as a tuple), so it can live in sets and be sorted for stable output.
  """Information about a single tf_export instance."""

  file_name: str
  line_no: int
  symbol_name: str
  v1_apis: tuple[str, ...]
  v2_apis: tuple[str, ...]

  @classmethod
  def create(
      cls, *, v1_apis: Sequence[str], v2_apis: Sequence[str], **kwargs
  ) -> "ExportedSymbol":
    # [FUNCTION DOC]
    # Alternative constructor (class method). Inputs: v1_apis and v2_apis, any sequences of
    # strings (such as lists read from JSON), plus the remaining fields as keyword arguments
    # (file name, line number, symbol name). Converts both API sequences to tuples so the record
    # is hashable and returns the new ExportedSymbol. Missing or extra keywords raise a type error.
    return cls(v1_apis=tuple(v1_apis), v2_apis=tuple(v2_apis), **kwargs)


class ExportedDoc(NamedTuple):
  # [CLASS DOC]
  # Immutable record (named tuple) describing a module docstring registered for export: source
  # file name, line number, a tuple of public module names it documents, and the docstring text.
  # Hashable and orderable, so it can be stored in sets and sorted.
  """Information about an export Module docstring."""

  file_name: str
  line_no: int
  modules: tuple[str, ...]
  docstring: str

  @classmethod
  def create(cls, *, modules: Sequence[str], **kwargs) -> "ExportedDoc":
    # [FUNCTION DOC]
    # Alternative constructor (class method). Input: modules, a sequence of module name strings
    # (for example a list read from JSON), plus the other fields (file name, line number,
    # docstring) as keyword arguments. Converts modules to a tuple and returns the new
    # ExportedDoc. Missing or extra keywords raise a type error.
    return cls(modules=tuple(modules), **kwargs)


class ExportedApi(object):
  # [CLASS DOC]
  # Mutable collection of exported docstrings and exported symbols, each kept in a set (so
  # duplicates collapse). Can be created empty or from iterables, extended with add methods,
  # serialized to a JSON file (sorted for deterministic output) and merged from JSON files.
  # Equality compares type, docs and symbols. Because equality is defined without a hash method,
  # instances are not hashable.
  """ExportedApi is a collection of ExportedSymbols."""

  _docs: set[ExportedDoc]
  _symbols: set[ExportedSymbol]

  def __init__(
      self,
      *,
      docs: Iterable[ExportedDoc] = (),
      symbols: Iterable[ExportedSymbol] = (),
  ):
    # [FUNCTION DOC]
    # Constructor (keyword-only arguments). Inputs: docs, an optional iterable of exported docs,
    # and symbols, an optional iterable of exported symbols (both default to empty). Copies each
    # into a new set held by the instance. Returns nothing; the inputs are not modified.
    self._docs = set(docs)
    self._symbols = set(symbols)

  def write(self, filename: str, **kwargs) -> None:
    # [FUNCTION DOC]
    # Serializes the collection to a JSON file. Inputs: filename (path, created or overwritten,
    # written as UTF-8) and extra keyword arguments passed through to the JSON writer (for
    # example indentation). The file holds an object with a list of docs and a list of symbols,
    # each sorted and converted to a dictionary of its fields. Returns nothing. File system errors
    # propagate.
    """Writes exports to filename."""
    with open(filename, mode="w", encoding="utf-8") as f:
      json.dump(
          {
              "docs": [d._asdict() for d in sorted(self.docs)],
              "symbols": [s._asdict() for s in sorted(self.symbols)],
          },
          f,
          **kwargs,
      )

  def read(self, filename: str) -> None:
    # [FUNCTION DOC]
    # Loads a JSON file in the format produced by write and merges its content into this
    # collection (existing entries are kept; duplicates collapse). Input: filename (UTF-8 JSON
    # path). Each doc and symbol dictionary is turned back into a record using the alternative
    # constructors. Returns nothing. Raises file errors, JSON decode errors, a key error if the
    # docs or symbols list is missing, or a type error for unexpected fields.
    """Reads exports from filename."""
    with open(filename, mode="r", encoding="utf-8") as f:
      data = json.load(f)
      self._docs.update(ExportedDoc.create(**d) for d in data["docs"])
      self._symbols.update(ExportedSymbol.create(**s) for s in data["symbols"])

  def add_symbol(self, export: ExportedSymbol) -> None:
    # [FUNCTION DOC]
    # Adds one exported symbol record to the collection (no effect if already present). Returns
    # nothing. Mutates the instance.
    self._symbols.add(export)

  def add_doc(self, export: ExportedDoc) -> None:
    # [FUNCTION DOC]
    # Adds one exported docstring record to the collection (no effect if already present).
    # Returns nothing. Mutates the instance.
    self._docs.add(export)

  @property
  def docs(self) -> Iterable[ExportedDoc]:
    # [FUNCTION DOC]
    # Read-only property returning the internal set of exported docs (the live set, not a copy,
    # so callers should not modify it).
    return self._docs

  @property
  def symbols(self) -> Iterable[ExportedSymbol]:
    # [FUNCTION DOC]
    # Read-only property returning the internal set of exported symbols (the live set, not a
    # copy, so callers should not modify it).
    return self._symbols

  def __str__(self) -> str:
    # [FUNCTION DOC]
    # Returns a JSON text of the collection with a sorted docs list and a sorted symbols list,
    # each record as a dictionary of fields (the same shape as the written file, but compact).
    return json.dumps({
        "docs": [d._asdict() for d in sorted(self.docs)],
        "symbols": [s._asdict() for s in sorted(self.symbols)],
    })

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # Returns the same JSON text as the string conversion, so debugging output and test failure
    # messages show the content.
    return str(self)

  def __eq__(self, o: Any) -> bool:
    # [FUNCTION DOC]
    # KNOWN ISSUE: defining equality without a hash method makes instances unhashable, and the
    # check uses an exact type match, so subclasses never compare equal to the base class.
    # Equality: true only if the other object has exactly the same type and both the docs and
    # symbols sets are equal. Input: o, any object. Returns a boolean.
    return (
        type(self) is type(o)
        and self.docs == o.docs
        and self.symbols == o.symbols
    )
