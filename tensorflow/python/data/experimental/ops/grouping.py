# [MODULE DOC]
# Grouping transformations for tf.data datasets (experimental namespace).
# Exposes group_by_reducer (group elements by key and fold each group with a Reducer),
# group_by_window and bucket_by_sequence_length (deprecated thin wrappers that delegate to the
# same-named methods of the Dataset class), and the Reducer class that bundles the three
# functions (initialize, reduce, finalize) used by group_by_reducer.
# Internally defines a private dataset class that wraps the three user functions in structured
# function wrappers and builds the underlying experimental group-by-reducer dataset operation.
# Import-time side effects: the public names are registered for export under the experimental
# data API through the export decorator; deprecation decorators are applied to two functions.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Grouping dataset transformations."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.group_by_reducer")
def group_by_reducer(key_func, reducer):
  # [FUNCTION DOC]
  # Public entry point. Inputs: key_func, a function mapping a dataset element to a scalar
  # 64-bit integer key; reducer, a Reducer holding initialize, reduce and finalize functions.
  # Returns a transformation function (to be passed to the dataset apply method) that wraps the
  # given dataset in the private group-by-reducer dataset. Nothing is validated until that
  # transformation is applied; validation errors come from the dataset constructor.
  """A transformation that groups elements and performs a reduction.

  This transformation maps element of a dataset to a key using `key_func` and
  groups the elements by key. The `reducer` is used to process each group; its
  `init_func` is used to initialize state for each group when it is created, the
  `reduce_func` is used to update the state every time an element is mapped to
  the matching group, and the `finalize_func` is used to map the final state to
  an output value.

  Args:
    key_func: A function mapping a nested structure of tensors
      (having shapes and types defined by `self.output_shapes` and
      `self.output_types`) to a scalar `tf.int64` tensor.
    reducer: An instance of `Reducer`, which captures the reduction logic using
      the `init_func`, `reduce_func`, and `finalize_func` functions.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    # [FUNCTION DOC]
    # HELPER: closure returned by group_by_reducer.
    # Takes an input dataset and returns a new dataset that groups and reduces it using the
    # captured key function and reducer. No other side effects.
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return _GroupByReducerDataset(dataset, key_func, reducer)

  return _apply_fn


@deprecation.deprecated(None, "Use `tf.data.Dataset.group_by_window(...)`.")
@tf_export("data.experimental.group_by_window")
def group_by_window(key_func,
                    reduce_func,
                    window_size=None,
                    window_size_func=None):
  # [FUNCTION DOC]
  # Deprecated wrapper (use the dataset method group_by_window instead). Inputs: key_func maps
  # an element to a scalar 64-bit integer key; reduce_func maps a key and a dataset of up to
  # window_size consecutive same-key elements to a dataset; window_size is a constant window
  # size and window_size_func maps a key to a window size (exactly one of the two should be
  # given, both default to nothing). Returns a transformation function that calls the dataset
  # method group_by_window with these arguments. The documented ValueError for missing or
  # duplicate window arguments is raised by that method when the transformation is applied,
  # not by this function.
  """A transformation that groups windows of elements by key and reduces them.

  This transformation maps each consecutive element in a dataset to a key
  using `key_func` and groups the elements by key. It then applies
  `reduce_func` to at most `window_size_func(key)` elements matching the same
  key. All except the final window for each key will contain
  `window_size_func(key)` elements; the final window may be smaller.

  You may provide either a constant `window_size` or a window size determined by
  the key through `window_size_func`.

  Args:
    key_func: A function mapping a nested structure of tensors
      (having shapes and types defined by `self.output_shapes` and
      `self.output_types`) to a scalar `tf.int64` tensor.
    reduce_func: A function mapping a key and a dataset of up to `window_size`
      consecutive elements matching that key to another dataset.
    window_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      consecutive elements matching the same key to combine in a single
      batch, which will be passed to `reduce_func`. Mutually exclusive with
      `window_size_func`.
    window_size_func: A function mapping a key to a `tf.int64` scalar
      `tf.Tensor`, representing the number of consecutive elements matching
      the same key to combine in a single batch, which will be passed to
      `reduce_func`. Mutually exclusive with `window_size`.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: if neither or both of {`window_size`, `window_size_func`} are
      passed.
  """

  def _apply_fn(dataset):
    # [FUNCTION DOC]
    # HELPER: closure returned by group_by_window.
    # Takes an input dataset and returns the result of calling its group_by_window method with
    # the captured key function, reduce function, window size and window size function.
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return dataset.group_by_window(
        key_func=key_func,
        reduce_func=reduce_func,
        window_size=window_size,
        window_size_func=window_size_func)

  return _apply_fn


@deprecation.deprecated(None,
                        "Use `tf.data.Dataset.bucket_by_sequence_length(...)`.")
@tf_export("data.experimental.bucket_by_sequence_length")
def bucket_by_sequence_length(element_length_func,
                              bucket_boundaries,
                              bucket_batch_sizes,
                              padded_shapes=None,
                              padding_values=None,
                              pad_to_bucket_boundary=False,
                              no_padding=False,
                              drop_remainder=False):
  # [FUNCTION DOC]
  # Deprecated wrapper (use the dataset method bucket_by_sequence_length instead). Inputs:
  # element_length_func maps an element to its length; bucket_boundaries is a list of upper
  # length boundaries; bucket_batch_sizes is a list of batch sizes, one per bucket (one more
  # than the number of boundaries); padded_shapes and padding_values optionally control
  # padding (default nothing); pad_to_bucket_boundary pads unknown dimensions to the bucket
  # boundary minus one; no_padding disables padding; drop_remainder drops a final short batch
  # (the last three default to false). Returns a transformation function delegating to the
  # dataset method with all arguments. Validation (the documented ValueError on mismatched
  # lengths) happens in that method when the transformation is applied.
  """A transformation that buckets elements in a `Dataset` by length.

  Elements of the `Dataset` are grouped together by length and then are padded
  and batched.

  This is useful for sequence tasks in which the elements have variable length.
  Grouping together elements that have similar lengths reduces the total
  fraction of padding in a batch which increases training step efficiency.

  Below is an example to bucketize the input data to the 3 buckets
  "[0, 3), [3, 5), [5, inf)" based on sequence length, with batch size 2.

  >>> elements = [
  ...   [0], [1, 2, 3, 4], [5, 6, 7],
  ...   [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]

  >>> dataset = tf.data.Dataset.from_generator(
  ...     lambda: elements, tf.int64, output_shapes=[None])

  >>> dataset = dataset.apply(
  ...     tf.data.experimental.bucket_by_sequence_length(
  ...         element_length_func=lambda elem: tf.shape(elem)[0],
  ...         bucket_boundaries=[3, 5],
  ...         bucket_batch_sizes=[2, 2, 2]))

  >>> for elem in dataset.as_numpy_iterator():
  ...   print(elem)
  [[1 2 3 4]
   [5 6 7 0]]
  [[ 7  8  9 10 11  0]
   [13 14 15 16 19 20]]
  [[ 0  0]
   [21 22]]

  There is also a possibility to pad the dataset till the bucket boundary.
  You can also provide which value to be used while padding the data.
  Below example uses `-1` as padding and it also shows the input data
  being bucketizied to two buckets "[0,3], [4,6]".

  >>> elements = [
  ...   [0], [1, 2, 3, 4], [5, 6, 7],
  ...   [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]

  >>> dataset = tf.data.Dataset.from_generator(
  ...   lambda: elements, tf.int32, output_shapes=[None])

  >>> dataset = dataset.apply(
  ...     tf.data.experimental.bucket_by_sequence_length(
  ...         element_length_func=lambda elem: tf.shape(elem)[0],
  ...         bucket_boundaries=[4, 7],
  ...         bucket_batch_sizes=[2, 2, 2],
  ...         pad_to_bucket_boundary=True,
  ...         padding_values=-1))

  >>> for elem in dataset.as_numpy_iterator():
  ...   print(elem)
  [[ 0 -1 -1]
   [ 5  6  7]]
  [[ 1  2  3  4 -1 -1]
   [ 7  8  9 10 11 -1]]
  [[21 22 -1]]
  [[13 14 15 16 19 20]]

  When using `pad_to_bucket_boundary` option, it can be seen that it is
  not always possible to maintain the bucket batch size.
  You can drop the batches that do not maintain the bucket batch size by
  using the option `drop_remainder`. Using the same input data as in the
  above example you get the following result.

  >>> elements = [
  ...   [0], [1, 2, 3, 4], [5, 6, 7],
  ...   [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]

  >>> dataset = tf.data.Dataset.from_generator(
  ...   lambda: elements, tf.int32, output_shapes=[None])

  >>> dataset = dataset.apply(
  ...     tf.data.experimental.bucket_by_sequence_length(
  ...         element_length_func=lambda elem: tf.shape(elem)[0],
  ...         bucket_boundaries=[4, 7],
  ...         bucket_batch_sizes=[2, 2, 2],
  ...         pad_to_bucket_boundary=True,
  ...         padding_values=-1,
  ...         drop_remainder=True))

  >>> for elem in dataset.as_numpy_iterator():
  ...   print(elem)
  [[ 0 -1 -1]
   [ 5  6  7]]
  [[ 1  2  3  4 -1 -1]
   [ 7  8  9 10 11 -1]]

  Args:
    element_length_func: function from element in `Dataset` to `tf.int32`,
      determines the length of the element, which will determine the bucket it
      goes into.
    bucket_boundaries: `list<int>`, upper length boundaries of the buckets.
    bucket_batch_sizes: `list<int>`, batch size per bucket. Length should be
      `len(bucket_boundaries) + 1`.
    padded_shapes: Nested structure of `tf.TensorShape` to pass to
      `tf.data.Dataset.padded_batch`. If not provided, will use
      `dataset.output_shapes`, which will result in variable length dimensions
      being padded out to the maximum length in each batch.
    padding_values: Values to pad with, passed to
      `tf.data.Dataset.padded_batch`. Defaults to padding with 0.
    pad_to_bucket_boundary: bool, if `False`, will pad dimensions with unknown
      size to maximum length in batch. If `True`, will pad dimensions with
      unknown size to bucket boundary minus 1 (i.e., the maximum length in each
      bucket), and caller must ensure that the source `Dataset` does not contain
      any elements with length longer than `max(bucket_boundaries)`.
    no_padding: `bool`, indicates whether to pad the batch features (features
      need to be either of type `tf.sparse.SparseTensor` or of same shape).
    drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
      whether the last batch should be dropped in the case it has fewer than
      `batch_size` elements; the default behavior is not to drop the smaller
      batch.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: if `len(bucket_batch_sizes) != len(bucket_boundaries) + 1`.
  """

  def _apply_fn(dataset):
    # [FUNCTION DOC]
    # HELPER: closure returned by bucket_by_sequence_length.
    # Takes an input dataset and returns the result of calling its bucket_by_sequence_length
    # method with all of the captured configuration arguments.
    return dataset.bucket_by_sequence_length(
        element_length_func=element_length_func,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        padded_shapes=padded_shapes,
        padding_values=padding_values,
        pad_to_bucket_boundary=pad_to_bucket_boundary,
        no_padding=no_padding,
        drop_remainder=drop_remainder)

  return _apply_fn


class _GroupByReducerDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # HELPER: private dataset implementation behind group_by_reducer.
  # A unary dataset that keeps the input dataset and four wrapped user functions (key,
  # initialize, reduce, finalize) and builds the experimental group-by-reducer dataset
  # operation from them. Also tracks the state structure of the reduction, which is widened
  # until a fixed point is reached. Its element structure is the output of the finalize function.
  """A `Dataset` that groups its input and performs a reduction."""

  def __init__(self, input_dataset, key_func, reducer):
    # [FUNCTION DOC]
    # Builds the dataset. Inputs: input_dataset, the dataset to group; key_func, function from an
    # element to a scalar 64-bit integer key; reducer, an object providing initialize, reduce
    # and finalize functions. Wraps each function (see the _make_* helpers, which store the
    # wrappers as attributes), then creates the variant tensor of the group-by-reducer
    # operation using each function and its captured inputs plus the flat structure of the
    # output, and calls the parent constructor. Raises ValueError or TypeError from the helpers
    # when functions have invalid output types, classes or shapes.
    """See `group_by_reducer()` for details."""
    self._input_dataset = input_dataset
    self._make_key_func(key_func, input_dataset)
    self._make_init_func(reducer.init_func)
    self._make_reduce_func(reducer.reduce_func, input_dataset)
    self._make_finalize_func(reducer.finalize_func)
    variant_tensor = ged_ops.experimental_group_by_reducer_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._key_func.function.captured_inputs,
        self._init_func.function.captured_inputs,
        self._reduce_func.function.captured_inputs,
        self._finalize_func.function.captured_inputs,
        key_func=self._key_func.function,
        init_func=self._init_func.function,
        reduce_func=self._reduce_func.function,
        finalize_func=self._finalize_func.function,
        **self._flat_structure)
    super(_GroupByReducerDataset, self).__init__(input_dataset, variant_tensor)

  def _make_key_func(self, key_func, input_dataset):
    # [FUNCTION DOC]
    # HELPER: wraps the user key function for the constructor.
    # Wraps key_func in a structured function wrapper bound to the input dataset's element
    # structure and stores it on the instance. Raises ValueError if the function's output is not
    # compatible with a scalar 64-bit integer tensor. Returns nothing.
    """Make wrapping defun for key_func."""
    self._key_func = structured_function.StructuredFunctionWrapper(
        key_func, self._transformation_name(), dataset=input_dataset)
    if not self._key_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.int64)):
      raise ValueError(
          f"Invalid `key_func`. Expected `key_func` to return a scalar "
          f"tf.int64 tensor, but instead `key_func` has output "
          f"types={self._key_func.output_types} "
          f"and shapes={self._key_func.output_shapes}."
      )

  def _make_init_func(self, init_func):
    # [FUNCTION DOC]
    # HELPER: wraps the user initialize function for the constructor.
    # Wraps init_func in a structured function wrapper whose input is a scalar 64-bit integer
    # (the group key) and stores it on the instance. Returns nothing.
    """Make wrapping defun for init_func."""
    self._init_func = structured_function.StructuredFunctionWrapper(
        init_func,
        self._transformation_name(),
        input_structure=tensor_spec.TensorSpec([], dtypes.int64))

  def _make_reduce_func(self, reduce_func, input_dataset):
    # [FUNCTION DOC]
    # HELPER: wraps the user reduce function and infers the state structure.
    # Starts from the structure produced by the initialize function and repeatedly wraps
    # reduce_func with inputs (state, element of input_dataset), checking that the output
    # classes and types match the state, and weakening the state shapes to the most specific
    # compatible shape. It repeats until the shapes stop changing (fixed point), updating the
    # state structure each round. Raises TypeError if output classes or element types differ
    # from the state. Stores the final wrapper and the state structure on the instance and adds
    # the function to the default graph. Returns nothing.
    # KNOWN ISSUE: the class-mismatch error message reads an attribute for the state classes
    # that is never set on the instance (only a local variable exists), so building that
    # message would itself fail with an attribute error instead of the intended TypeError.
    """Make wrapping defun for reduce_func."""

    # Iteratively rerun the reduce function until reaching a fixed point on
    # `self._state_structure`.
    self._state_structure = self._init_func.output_structure
    state_types = self._init_func.output_types
    state_shapes = self._init_func.output_shapes
    state_classes = self._init_func.output_classes
    need_to_rerun = True
    while need_to_rerun:

      wrapped_func = structured_function.StructuredFunctionWrapper(
          reduce_func,
          self._transformation_name(),
          input_structure=(self._state_structure, input_dataset.element_spec),
          add_to_graph=False)

      # Extract and validate class information from the returned values.
      for new_state_class, state_class in zip(
          nest.flatten(wrapped_func.output_classes),
          nest.flatten(state_classes)):
        if not issubclass(new_state_class, state_class):
          raise TypeError(
              f"Invalid `reducer`. The output class of the "
              f"`reducer.reduce_func` {wrapped_func.output_classes}, "
              f"does not match the class of the reduce state "
              f"{self._state_classes}.")

      # Extract and validate type information from the returned values.
      for new_state_type, state_type in zip(
          nest.flatten(wrapped_func.output_types), nest.flatten(state_types)):
        if new_state_type != state_type:
          raise TypeError(
              f"Invalid `reducer`. The element types for the new state "
              f"{wrapped_func.output_types} do not match the element types "
              f"of the old state {self._init_func.output_types}."
          )

      # Extract shape information from the returned values.
      flat_state_shapes = nest.flatten(state_shapes)
      flat_new_state_shapes = nest.flatten(wrapped_func.output_shapes)
      weakened_state_shapes = [
          original.most_specific_compatible_shape(new)
          for original, new in zip(flat_state_shapes, flat_new_state_shapes)
      ]

      need_to_rerun = False
      for original_shape, weakened_shape in zip(flat_state_shapes,
                                                weakened_state_shapes):
        if original_shape.ndims is not None and (
            weakened_shape.ndims is None or
            original_shape.as_list() != weakened_shape.as_list()):
          need_to_rerun = True
          break

      if need_to_rerun:
        state_shapes = nest.pack_sequence_as(
            self._init_func.output_shapes, weakened_state_shapes)
        self._state_structure = structure.convert_legacy_structure(
            state_types, state_shapes, state_classes)

    self._reduce_func = wrapped_func
    self._reduce_func.function.add_to_graph(ops.get_default_graph())

  def _make_finalize_func(self, finalize_func):
    # [FUNCTION DOC]
    # HELPER: wraps the user finalize function for the constructor.
    # Wraps finalize_func in a structured function wrapper whose input is the final state
    # structure computed by the reduce step and stores it on the instance. Returns nothing.
    """Make wrapping defun for finalize_func."""
    self._finalize_func = structured_function.StructuredFunctionWrapper(
        finalize_func,
        self._transformation_name(),
        input_structure=self._state_structure)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property giving the structure of dataset elements: the output structure of the wrapped
    # finalize function.
    return self._finalize_func.output_structure

  def _functions(self):
    # [FUNCTION DOC]
    # HELPER: returns the list of the four wrapped functions (key, initialize, reduce, finalize)
    # so the dataset framework can track them.
    return [
        self._key_func, self._init_func, self._reduce_func, self._finalize_func
    ]

  def _transformation_name(self):
    # [FUNCTION DOC]
    # HELPER: returns the display name of this transformation, used in error messages and
    # function naming.
    return "tf.data.experimental.group_by_reducer()"


@tf_export("data.experimental.Reducer")
class Reducer:
  # [CLASS DOC]
  # Plain container for the three functions that define a reduction: initialize (key to initial
  # state), reduce (old state and input element to new state) and finalize (final state to
  # result). Instances are passed to group_by_reducer. Holds the functions as private
  # attributes exposed through read-only properties; performs no validation.
  """A reducer is used for reducing a set of elements.

  A reducer is represented as a tuple of the three functions:
  - init_func - to define initial value: key => initial state
  - reducer_func - operation to perform on values with same key: (old state, input) => new state
  - finalize_func - value to return in the end: state => result
  
  For example,
  
  ```
  def init_func(_):
    return (0.0, 0.0)

  def reduce_func(state, value):
    return (state[0] + value['features'], state[1] + 1)

  def finalize_func(s, n):
    return s / n

  reducer = tf.data.experimental.Reducer(init_func, reduce_func, finalize_func)
  ```
  """

  def __init__(self, init_func, reduce_func, finalize_func):
    # [FUNCTION DOC]
    # Stores the three given functions (initialize, reduce, finalize) on the instance without
    # checking them. Returns nothing.
    self._init_func = init_func
    self._reduce_func = reduce_func
    self._finalize_func = finalize_func

  @property
  def init_func(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored initialize function.
    return self._init_func

  @property
  def reduce_func(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored reduce function.
    return self._reduce_func

  @property
  def finalize_func(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored finalize function.
    return self._finalize_func
