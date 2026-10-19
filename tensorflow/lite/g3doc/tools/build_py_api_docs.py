# [MODULE DOC]
# HELPER: Documentation build tool (command line script) for the tf.lite Python reference docs.
# Command line script that generates the Python API reference docs for the tf.lite module of
# TensorFlow using the external docs generator library. It defines flags for the output
# directory, code URL prefix, search hints and site path.
# Import time side effects: imports the full TensorFlow package and registers absl flags.
# Running it as a program starts the absl app with the main function.
# KNOWN ISSUE: the module docstring tells users to run a script named build_docs, but this file
# is named build_py_api_docs.
# Copyright 2024 The AI Edge Authors. All Rights Reserved.
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
r"""Generate python docs for tf.lite.

# How to run

```
python build_docs.py --output_dir=/path/to/output
```

"""

import pathlib

from absl import app
from absl import flags

import tensorflow as tf

from tensorflow_docs.api_generator import generate_lib

flags.DEFINE_string('output_dir', '/tmp/lite_api/',
                    'The path to output the files to')

flags.DEFINE_string('code_url_prefix',
                    'https://github.com/tensorflow/tensorflow/blob/master/',
                    'The url prefix for links to code.')

flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files')

flags.DEFINE_string('site_path', 'lite/api_docs/python',
                    'Path prefix in the _toc.yaml')

FLAGS = flags.FLAGS


def main(_):
  # [FUNCTION DOC]
  # Entry point of the Python docs build; command line arguments are ignored.
  # Builds a documentation generator titled TensorFlow Lite for the tf.lite module, with the base
  # directory being the installed TensorFlow package directory, and the code URL prefix, search
  # hints and site path taken from flags, with no extra callbacks.
  # Then builds the docs into the output directory flag. Returns nothing; writes files to disk.
  doc_generator = generate_lib.DocGenerator(
      root_title='TensorFlow Lite',
      py_modules=[('tf.lite', tf.lite)],
      base_dir=str(pathlib.Path(tf.__file__).parent),
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      callbacks=[])

  doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
