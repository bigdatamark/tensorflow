# [MODULE DOC]
# Command-line tool that prints the names of functions a stub library should forward for a
# given shared library. It loads the third-party implib-gen module (imported by name through the
# import machinery because the file name contains a dash), asks it to collect the symbol table
# of the library, keeps only exported default-version defined functions, and prints their
# names sorted, one per line. The output is meant to be saved as a symbols file consumed by the
# stub generator next to it. Import-time side effect: imports implib-gen, failing if absent.
"""Given a .so file, lists symbols that should be included in a stub.

Example usage:
$ bazel run -c opt @xla//third_party/implib_so:get_symbols
/usr/local/cuda/lib64/libcudart.so > third_party/tsl/tsl/cuda/cudart.symbols
"""

import argparse
import importlib

# We can't import implib-gen directly because it has a dash in its name.
implib = importlib.import_module('implib-gen')


def _is_exported_function(s):
  # [FUNCTION DOC]
  # HELPER: symbol filter used by the symbol listing entry point.
  # Input: one symbol record (mapping with binding, type, section index, name and default flag
  # fields) from the symbol table. Returns true when the symbol is non-local, is a function,
  # is defined (not undefined), has a name other than empty or the init/fini hooks, and is the
  # default version. Otherwise false. Pure function.
  return (
      s['Bind'] != 'LOCAL'
      and s['Type'] == 'FUNC'
      and s['Ndx'] != 'UND'
      and s['Name'] not in ['', '_init', '_fini']
      and s['Default']
  )


def main():
  # [FUNCTION DOC]
  # Command-line entry point. Takes one required positional argument, the path of the shared
  # library. Collects its symbols, filters to exported functions and prints the names, sorted
  # alphabetically, one per line on standard output. Returns nothing. File or parsing errors from
  # the symbol collection propagate; invalid arguments make the argument parser exit.
  parser = argparse.ArgumentParser(
      description='Extracts a list of symbols from a shared library'
  )
  parser.add_argument('library', help='Path to the .so file.')
  args = parser.parse_args()
  syms = implib.collect_syms(args.library)
  funs = [s['Name'] for s in syms if _is_exported_function(s)]
  for f in sorted(funs):
    print(f)


if __name__ == '__main__':
  main()
