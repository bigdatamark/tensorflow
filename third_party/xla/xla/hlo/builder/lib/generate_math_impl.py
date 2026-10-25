# [MODULE DOC]
# HELPER: developer script (not part of runtime code) that generates the C++ header math_impl.h.
# KNOWN ISSUE: the original docstring says the output is math_impl.cc, but the script writes
# math_impl.h.
# It produces XLA builder implementations of complex and real arcsine (functions named
# AsinComplex and AsinReal) using the external functional_algorithms package. It traces the library's algorithms,
# fills in missing pieces for the XLA client target, simplifies the graph, renders it as C++ code
# with the algorithm docstring as a comment, and writes the header, with license text, a note that
# it is generated, include guard and namespaces, next to this script.
# Import-time side effects: tries to import the optional functional_algorithms package; on failure
# only emits a warning and leaves the module reference empty, after which the main function does
# nothing. Running as a program calls main.
"""A script to generate math_impl.h.

Prerequisites:
  python 3.11 or newer
  functional_algorithms 0.3.1 or newer

Usage:
  Running
    python /path/to/generate_math_impl.py [xla | tensorflow]
  will create
    /path/to/math_impl.cc
"""

import os
import sys
import warnings

try:
  import functional_algorithms as fa  # pylint: disable=g-import-not-at-top
except ImportError as msg:
  warnings.warn(f"Skipping: {msg}")
  fa = None


def main():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the up-to-date check ends with a literal zero in its condition, so it never
  # triggers and the file is always regenerated (dead code); the file is read and written with
  # manual open and close calls that leak the handle if an error occurs.
  # KNOWN ISSUE: the variable named target first holds the chosen output flavor text and is then
  # overwritten by the functional_algorithms code generation target, which is confusing; and if
  # the first algorithm has no docstring, appending to the sources list fails with an index error.
  # Entry point. Takes no parameters; reads an optional first command-line argument, either xla
  # (default) or tensorflow, case-insensitively, which only selects the include guard macro name
  # (an assertion error is raised for any other value). If the optional library is missing it
  # returns at once. If its version is older than 0.3.1 it warns and returns. Then, for each of
  # the two arcsine algorithms, looks the algorithm up in the library (warning and skipping it
  # if absent), traces it into a graph with the XLA client target, simplifies it, names it, renders
  # it as C++ and collects it with the algorithm docstring as a leading comment. Joins the pieces,
  # and overwrites math_impl.h in the script's directory with the license header, generated-file
  # notice, include guard, includes, namespaces, formatting-off markers and the generated source.
  # Returns nothing; reports the created path through a warning.
  if fa is None:
    return
  target = (sys.argv[1] if len(sys.argv) > 1 else "xla").lower()
  assert target in {"xla", "tensorflow"}, target
  header_file_define = dict(
      xla="XLA_CLIENT_MATH_IMPL_H_",
      tensorflow="TENSORFLOW_COMPILER_XLA_CLIENT_MATH_IMPL_H_",
  )[target]

  fa_version = tuple(map(int, fa.__version__.split(".", 4)[:3]))
  if fa_version < (0, 3, 1):
    warnings.warn(
        "functional_algorithm version 0.3.1 or newer is required,"
        f" got {fa.__version__}"
    )
    return

  output_file = os.path.join(os.path.dirname(__file__), "math_impl.h")

  sources = []
  target = fa.targets.xla_client
  for xlaname, fname, args in [
      ("AsinComplex", "complex_asin", ("z:complex",)),
      ("AsinReal", "real_asin", ("x:float",)),
  ]:
    func = getattr(fa.algorithms, fname, None)
    if func is None:
      warnings.warn(
          f"{fa.algorithms.__name__} does not define {fname}. Skipping."
      )
      continue
    ctx = fa.Context(
        paths=[fa.algorithms],
        enable_alt=True,
        default_constant_type="FloatType",
    )
    graph = ctx.trace(func, *args).implement_missing(target).simplify()
    graph.props.update(name=xlaname)
    src = graph.tostring(target)
    if func.__doc__:
      sources.append(target.make_comment(func.__doc__))
    sources[-1] += src
  source = "\n\n".join(sources) + "\n"

  if os.path.isfile(output_file):
    f = open(output_file, "r", encoding="UTF-8")
    content = f.read()
    f.close()
    if content.endswith(source) and 0:
      warnings.warn(f"{output_file} is up-to-date.")
      return

  f = open(output_file, "w", encoding="UTF-8")
  f.write("""/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

""")
  f.write(target.make_comment(f"""\
This file is generated using functional_algorithms tool ({fa.__version__}), see
  https://github.com/pearu/functional_algorithms
for more information.""") + "\n")
  f.write(f"""\
#ifndef {header_file_define}
#define {header_file_define}

#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"

namespace xla {{
namespace math_impl {{
// NOLINTBEGIN(whitespace/line_length)
// clang-format off

""")
  f.write(source)
  f.write(f"""
// clang-format on
// NOLINTEND(whitespace/line_length)
}}  // namespace math_impl
}}  // namespace xla

#endif  // {header_file_define}
""")
  f.close()
  warnings.warn(f"Created {output_file}")


if __name__ == "__main__":
  main()
