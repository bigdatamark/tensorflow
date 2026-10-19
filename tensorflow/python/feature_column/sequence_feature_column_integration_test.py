# [MODULE DOC]
# Integration test module checking that sequence categorical feature columns work with the
# single sequence example parser. It builds a sequence example from a text protocol buffer
# constant containing context features plus bytes, float and integer feature lists, parses it with
# specs derived from the columns, and checks the parsed sparse shapes and values. Has no
# import-time side effects beyond defining the constant; runs via the test main entry point.
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
"""Integration test for sequence feature columns with SequenceExamples."""

import string
import tempfile

from google.protobuf import text_format

from tensorflow.core.example import example_pb2
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import sequence_feature_column as sfc
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class SequenceExampleParsingTest(test.TestCase):
  # [CLASS DOC]
  # Tests that each sequence categorical column type (identity, hash bucket, vocabulary list,
  # vocabulary file) produces parse specs that parse correctly from a single sequence example.
  # Shared fixture: the module-level sequence example built from a text proto; shared helper
  # performs the parsing and assertions.

  def test_seq_ex_in_sequence_categorical_column_with_identity(self):
    # [FUNCTION DOC]
    # Scenario: identity column on the integer feature list with ten buckets. Expects parsed sparse
    # dense shape (3, 6) and values 2, 4, 6 at
    # flat indices 0, 2 and 6. Delegates to the shared helper.
    self._test_parsed_sequence_example(
        'int_list', sfc.sequence_categorical_column_with_identity,
        10, [3, 6], [2, 4, 6])

  def test_seq_ex_in_sequence_categorical_column_with_hash_bucket(self):
    # [FUNCTION DOC]
    # Scenario: hash bucket column of size ten on the bytes feature list. Expects dense shape
    # (3, 4) and the byte strings a, c, g at flat indices 0, 2 and 6. Delegates to the shared helper.
    self._test_parsed_sequence_example(
        'bytes_list', sfc.sequence_categorical_column_with_hash_bucket,
        10, [3, 4], [compat.as_bytes(x) for x in 'acg'])

  def test_seq_ex_in_sequence_categorical_column_with_vocabulary_list(self):
    # [FUNCTION DOC]
    # Scenario: vocabulary-list column over the lowercase alphabet on the bytes feature list.
    # Expects dense shape (3, 4) and byte strings a, c, g at flat indices 0, 2 and 6. Raw values are
    # parsed (not mapped to ids) by the parse spec. Delegates to the shared helper.
    self._test_parsed_sequence_example(
        'bytes_list', sfc.sequence_categorical_column_with_vocabulary_list,
        list(string.ascii_lowercase), [3, 4],
        [compat.as_bytes(x) for x in 'acg'])

  def test_seq_ex_in_sequence_categorical_column_with_vocabulary_file(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the temporary file and its open descriptor are leaked after the test.
    # Scenario: writes the lowercase alphabet to a temporary file (side effect: the file is created
    # and never removed; the file descriptor from creation is not closed), then uses a
    # vocabulary-file column on the bytes feature list. Expects dense shape (3, 4) and byte strings
    # a, c, g at flat indices 0, 2 and 6. Delegates to the shared helper.
    _, fname = tempfile.mkstemp()
    with open(fname, 'w') as f:
      f.write(string.ascii_lowercase)
    self._test_parsed_sequence_example(
        'bytes_list', sfc.sequence_categorical_column_with_vocabulary_file,
        fname, [3, 4], [compat.as_bytes(x) for x in 'acg'])

  def _test_parsed_sequence_example(
      self, col_name, col_fn, col_arg, shape, values):
    # [FUNCTION DOC]
    # HELPER: shared check used by the parsing tests in this class.
    # Inputs: feature name, a column factory, its second argument, the expected sparse dense shape
    # and the expected values at flat indices 0, 2 and 6. Builds the sequence example, creates an
    # identity context column, a numeric context column and the sequence column, parses the
    # example with specs from those columns, runs the result in a cached session and asserts the
    # sequence shape and values plus the context values (integer 5, float about 123.6).
    """Helper function to check that each FeatureColumn parses correctly.

    Args:
      col_name: string, name to give to the feature column. Should match
        the name that the column will parse out of the features dict.
      col_fn: function used to create the feature column. For example,
        sequence_numeric_column.
      col_arg: second arg that the target feature column is expecting.
      shape: the expected dense_shape of the feature after parsing into
        a SparseTensor.
      values: the expected values at index [0, 2, 6] of the feature
        after parsing into a SparseTensor.
    """
    example = _make_sequence_example()
    columns = [
        fc.categorical_column_with_identity('int_ctx', num_buckets=100),
        fc.numeric_column('float_ctx'),
        col_fn(col_name, col_arg)
    ]
    context, seq_features = parsing_ops.parse_single_sequence_example(
        example.SerializeToString(),
        context_features=fc.make_parse_example_spec_v2(columns[:2]),
        sequence_features=fc.make_parse_example_spec_v2(columns[2:]))

    with self.cached_session() as sess:
      ctx_result, seq_result = sess.run([context, seq_features])
      self.assertEqual(list(seq_result[col_name].dense_shape), shape)
      self.assertEqual(
          list(seq_result[col_name].values[[0, 2, 6]]), values)
      self.assertEqual(list(ctx_result['int_ctx'].dense_shape), [1])
      self.assertEqual(ctx_result['int_ctx'].values[0], 5)
      self.assertEqual(list(ctx_result['float_ctx'].shape), [1])
      self.assertAlmostEqual(ctx_result['float_ctx'][0], 123.6, places=1)


_SEQ_EX_PROTO = """
context {
  feature {
    key: "float_ctx"
    value {
      float_list {
        value: 123.6
      }
    }
  }
  feature {
    key: "int_ctx"
    value {
      int64_list {
        value: 5
      }
    }
  }
}
feature_lists {
  feature_list {
    key: "bytes_list"
    value {
      feature {
        bytes_list {
          value: "a"
        }
      }
      feature {
        bytes_list {
          value: "b"
          value: "c"
        }
      }
      feature {
        bytes_list {
          value: "d"
          value: "e"
          value: "f"
          value: "g"
        }
      }
    }
  }
  feature_list {
    key: "float_list"
    value {
      feature {
        float_list {
          value: 1.0
        }
      }
      feature {
        float_list {
          value: 3.0
          value: 3.0
          value: 3.0
        }
      }
      feature {
        float_list {
          value: 5.0
          value: 5.0
          value: 5.0
          value: 5.0
          value: 5.0
        }
      }
    }
  }
  feature_list {
    key: "int_list"
    value {
      feature {
        int64_list {
          value: 2
          value: 2
        }
      }
      feature {
        int64_list {
          value: 4
          value: 4
          value: 4
          value: 4
        }
      }
      feature {
        int64_list {
          value: 6
          value: 6
          value: 6
          value: 6
          value: 6
          value: 6
        }
      }
    }
  }
}
"""


def _make_sequence_example():
  # [FUNCTION DOC]
  # HELPER: test fixture builder. Returns a new sequence example message parsed from the text
  # proto constant defined in this module. Raises a parse error if the text were malformed.
  example = example_pb2.SequenceExample()
  return text_format.Parse(_SEQ_EX_PROTO, example)


if __name__ == '__main__':
  test.main()
