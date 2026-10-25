# [MODULE DOC]
# Helper module for the TensorFlow API documentation generator: decides, for the installed
# TensorFlow, which source directories the doc generator should search for symbol definitions and
# which code-link URL prefix belongs to each directory (TensorFlow itself, Keras, TensorBoard and,
# for older versions, the estimator package). Also provides the filter callback that keeps the
# keras symbol in the public API tree. Import-time side effects: imports keras, tensorboard,
# tensorflow and the docs tooling; the estimator package is optional and becomes null if absent.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Opensource base_dir configuration for tensorflow doc-generator."""
import pathlib

import keras
from packaging import version
import tensorboard
import tensorflow as tf
from tensorflow_docs.api_generator import public_api

try:
  import tensorflow_estimator  # pylint: disable=[g-import-not-at-top, g-deprecated-tf-checker]
except ImportError:
  tensorflow_estimator = None


def get_base_dirs_and_prefixes(code_url_prefix):
  # [FUNCTION DOC]
  # HELPER: configuration glue for the documentation generator.
  # Input: code_url_prefix, the URL prefix to use for links into TensorFlow's own source.
  # Output: a pair of a list of base directories (ordered; the first match wins and unmatched
  # objects are dropped, with the TensorFlow directory last) and a tuple of URL prefixes in the
  # same order, where a null entry means do not link. The layout depends on the installed
  # TensorFlow version (16 and later, 13 and later, 9 and later). Keras links use the master branch
  # for development builds and the matching release tag otherwise.
  # Raises a value error for versions older than 2.9.
  # KNOWN ISSUE: the 2.9 to 2.15 branches dereference the estimator package, which is null when it
  # is not installed, so they would fail with an attribute error in that case.
  """Returns the base_dirs and code_prefixes for OSS TensorFlow api gen."""
  base_dir = pathlib.Path(tf.__file__).parent

  if "dev" in tf.__version__:
    keras_url_prefix = "https://github.com/keras-team/keras/tree/master/keras/src"
  else:
    keras_url_prefix = (
        f"https://github.com/keras-team/keras/tree/v{keras.__version__}/keras/src"
    )

  if version.parse(tf.__version__) >= version.parse("2.16"):
    # First match takes precedence.
    # Objects are dropped if they have no match.
    base_dirs = [
        # The real keras source files are now in `site-packages/keras/src/...`
        pathlib.Path(keras.__file__).parent / "src",
        # The generated module files in tensorflow are in keras
        # under `site-packages/keras/api/_v2/keras/...`.
        pathlib.Path(tf.keras.__file__).parent,
        # The generated api-module files are now in `site-packages/keras/...`
        pathlib.Path(keras.__file__).parent,
        pathlib.Path(tensorboard.__file__).parent,
        # The tensorflow base dir goes last because `tf.keras``
        base_dir,
    ]

    code_url_prefixes = (
        keras_url_prefix,
        # None -> don't link to the generated keras api-module files.
        None,
        None,
        f"https://github.com/tensorflow/tensorboard/tree/{tensorboard.__version__}/tensorboard",
        code_url_prefix,
    )

  elif version.parse(tf.__version__) >= version.parse("2.13"):
    # First match takes precedence.
    # Objects are dropped if they have no match.
    base_dirs = [
        # The real keras source files are now in `site-packages/keras/src/...`
        pathlib.Path(keras.__file__).parent / "src",
        # The generated module files in tensorflow are in keras
        # under `site-packages/keras/api/_v2/keras/...`.
        pathlib.Path(tf.keras.__file__).parent,
        # The generated api-module files are now in `site-packages/keras/...`
        pathlib.Path(keras.__file__).parent,
        pathlib.Path(tensorboard.__file__).parent,
        pathlib.Path(tensorflow_estimator.__file__).parent,
        # The tensorflow base dir goes last because `tf.keras``
        base_dir,
    ]

    code_url_prefixes = (
        keras_url_prefix,
        # None -> don't link to the generated keras api-module files.
        None,
        None,
        f"https://github.com/tensorflow/tensorboard/tree/{tensorboard.__version__}/tensorboard",
        "https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator",
        code_url_prefix,
    )
  elif version.parse(tf.__version__) >= version.parse("2.9"):
    base_dirs = [
        base_dir,
        pathlib.Path(keras.__file__).parent,
        pathlib.Path(tensorboard.__file__).parent,
        pathlib.Path(tensorflow_estimator.__file__).parent,
    ]
    code_url_prefixes = (
        code_url_prefix,
        keras_url_prefix,
        f"https://github.com/tensorflow/tensorboard/tree/{tensorboard.__version__}/tensorboard",
        "https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator",
    )
  else:
    raise ValueError("Unsupported: version < 2.9")

  return base_dirs, code_url_prefixes


def explicit_filter_keep_keras(parent_path, parent, children):
  # [FUNCTION DOC]
  # HELPER: public-API filter callback for the doc generator.
  # Inputs: parent_path (list of names from the root), parent (the module or class) and children
  # (list of name and object pairs). Applies the standard explicit package contents filter, and
  # when the parent is the tf, v1 or v2 package and the filter dropped a keras child, appends it
  # back from the parent. Returns the filtered children, sorted by name when the keras check ran,
  # otherwise the filter result unchanged. No side effects.
  """Like explicit_package_contents_filter, but keeps keras."""
  new_children = public_api.explicit_package_contents_filter(
      parent_path, parent, children)

  if parent_path[-1] not in ["tf", "v1", "v2"]:
    return new_children

  had_keras = any(name == "keras" for name, child in children)
  has_keras = any(name == "keras" for name, child in new_children)

  if had_keras and not has_keras:
    new_children.append(("keras", parent.keras))

  return sorted(new_children, key=lambda x: x[0])


def get_callbacks():
  # [FUNCTION DOC]
  # HELPER: returns the list of filter callbacks (just the keep-keras filter) for the doc generator.
  return [explicit_filter_keep_keras]
