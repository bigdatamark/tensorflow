# [MODULE DOC]
# Command-line tool that generates the source files for a lazy-loading stub of a shared library
# (used for CUDA libraries). From a symbols file it writes an assembly file containing a table of
# resolved-symbol slots followed by one trampoline per symbol, and an include file listing the
# symbol names as quoted, comma-terminated lines. Architecture-specific configuration and
# assembly templates are located through the Bazel runfiles of the vendored implib package.
# Import-time side effect: creates a runfiles lookup object.
"""Given a list of symbols, generates a stub."""

import argparse
import configparser
import os
import string

from python.runfiles import runfiles

r = runfiles.Create()


def main():
  # [FUNCTION DOC]
  # Command-line entry point. Arguments: a required positional symbols file (one symbol name per
  # line), an optional output directory (default current directory) and a required target
  # architecture name. Reads the architecture's configuration file to get the pointer size, then
  # the table and trampoline templates. The library name is the symbols file name without its
  # extension. Writes an assembly file named after the library that holds the table (sized for
  # one slot per symbol plus one) followed by a trampoline for each symbol with its slot offset
  # and index, and an include file with each symbol name as a quoted line. Returns nothing.
  # Missing files raise file errors; bad arguments make the argument parser exit.
  parser = argparse.ArgumentParser(
      description='Generates stubs for CUDA libraries.'
  )
  parser.add_argument('symbols', help='File containing a list of symbols.')
  parser.add_argument(
      '--outdir', '-o', help='Path to create wrapper at', default='.'
  )
  parser.add_argument(
      '--target',
      help='Target platform name, e.g. x86_64, aarch64.',
      required=True,
  )
  args = parser.parse_args()

  config_path = r.Rlocation(f'implib_so/arch/{args.target}/config.ini')
  table_path = r.Rlocation(f'implib_so/arch/{args.target}/table.S.tpl')
  trampoline_path = r.Rlocation(
      f'implib_so/arch/{args.target}/trampoline.S.tpl'
  )

  cfg = configparser.ConfigParser(inline_comment_prefixes=';')
  cfg.read(config_path)
  ptr_size = int(cfg['Arch']['PointerSize'])

  with open(args.symbols, 'r') as f:
    funs = [s.strip() for s in f.readlines()]

  # Generate assembly code, containing a table for the resolved symbols and the
  # trampolines.
  lib_name, _ = os.path.splitext(os.path.basename(args.symbols))

  with open(os.path.join(args.outdir, f'{lib_name}.tramp.S'), 'w') as f:
    with open(table_path, 'r') as t:
      table_text = string.Template(t.read()).substitute(
          lib_suffix=lib_name, table_size=ptr_size * (len(funs) + 1)
      )
    f.write(table_text)

    with open(trampoline_path, 'r') as t:
      tramp_tpl = string.Template(t.read())

    for i, name in enumerate(funs):
      tramp_text = tramp_tpl.substitute(
          lib_suffix=lib_name, sym=name, offset=i * ptr_size, number=i
      )
      f.write(tramp_text)

  # Generates a list of symbols, formatted as a list of C++ strings.
  with open(os.path.join(args.outdir, f'{lib_name}.inc'), 'w') as f:
    sym_names = ''.join(f'  "{name}",\n' for name in funs)
    f.write(sym_names)


if __name__ == '__main__':
  main()
