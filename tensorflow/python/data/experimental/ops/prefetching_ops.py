# [MODULE DOC]
# Experimental device-prefetching transformations for tf.data. Exposes prefetch_to_device (copy
# to a device then prefetch), copy_to_device (copy elements to a target device by running the
# input pipeline on a source device through remote calls inside a generator dataset), and the
# highly experimental map_on_gpu (map function executed on GPU). Private dataset classes
# implement the copy and GPU map transformations.
# Import-time side effects: prefetch_to_device and copy_to_device are registered for export
# under the experimental data API; map_on_gpu is not exported.
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
"""Python wrapper for prefetching_ops."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import structure
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as framework_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.prefetch_to_device")
def prefetch_to_device(device, buffer_size=None):
  # [FUNCTION DOC]
  # Creates a transformation that copies elements to a device and prefetches them there. Inputs:
  # device, the target device name string; buffer_size, optional number of elements to buffer
  # (default nothing, meaning automatic). Returns a function to apply to a dataset. The result
  # should be the last stage of an input pipeline.
  """A transformation that prefetches dataset values to the given `device`.

  NOTE: Although the transformation creates a `tf.data.Dataset`, the
  transformation must be the final `Dataset` in the input pipeline.

  For example,
  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/cpu:0"))
  >>> for element in dataset:
  ...   print(f'Tensor {element} is on device {element.device}')
  Tensor 1 is on device /job:localhost/replica:0/task:0/device:CPU:0
  Tensor 2 is on device /job:localhost/replica:0/task:0/device:CPU:0
  Tensor 3 is on device /job:localhost/replica:0/task:0/device:CPU:0

  Args:
    device: A string. The name of a device to which elements will be prefetched.
    buffer_size: (Optional.) The number of elements to buffer on `device`.
      Defaults to an automatically chosen value.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    # [FUNCTION DOC]
    # HELPER: closure returned by prefetch_to_device.
    # Applies the copy-to-device transformation for the captured device, then a prefetch with
    # the captured buffer size, and returns the resulting dataset.
    return dataset.apply(
        copy_to_device(target_device=device)).prefetch(buffer_size)

  return _apply_fn


@tf_export("data.experimental.copy_to_device")
def copy_to_device(target_device, source_device="/cpu:0"):
  # [FUNCTION DOC]
  # Creates a transformation that copies dataset elements to another device. Inputs:
  # target_device, name of the device receiving elements; source_device, device where the input
  # dataset is placed (default the first CPU). Returns a function to apply to a dataset that
  # wraps it in the private copy-to-device dataset.
  """A transformation that copies dataset elements to the given `target_device`.

  Args:
    target_device: The name of a device to which elements will be copied.
    source_device: The original device on which `input_dataset` will be placed.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    # [FUNCTION DOC]
    # HELPER: closure returned by copy_to_device.
    # Takes a dataset and returns the private copy-to-device dataset built from it with the
    # captured target and source devices.
    return _CopyToDeviceDataset(
        dataset, target_device=target_device, source_device=source_device)

  return _apply_fn


# TODO(rohanj): Use the _input_hostmem attr on the RemoteCall ops to indicate
# all inputs to the Op are in host memory, thereby avoiding some unnecessary
# Sends and Recvs.
class _CopyToDeviceDataset(dataset_ops.UnaryUnchangedStructureDataset):
  # [CLASS DOC]
  # HELPER: private dataset implementation behind copy_to_device.
  # Unary dataset with unchanged element structure. It wraps the input dataset as a variant,
  # defines traced init, next and finalize functions that create, advance and destroy an
  # iterator on the source device (invoked through remote calls), registers them in the default
  # graph, and builds a generator dataset placed on the target device that drives them.
  # Remembers the target device, whether it is a GPU, and the source device.
  """A `Dataset` that copies elements to another device."""

  def __init__(self, input_dataset, target_device, source_device="/cpu:0"):
    # [FUNCTION DOC]
    # Builds the dataset. Inputs: input_dataset, the dataset to copy (debug options applied);
    # target_device, device name receiving elements; source_device, device name where the input
    # runs (default first CPU). Stores device information (including whether the target is a
    # GPU), wraps the input as a variant, defines the nested init, next and finalize traced
    # functions with their remote-call wrappers, adds the three concrete functions to the
    # default graph, and creates the generator dataset on the target device. Calls the parent
    # constructor. Returns nothing; touches graph state and devices.
    """Constructs a _CopyToDeviceDataset.

    Args:
      input_dataset: `Dataset` to be copied
      target_device: The name of the device to which elements would be copied.
      source_device: Device where input_dataset would be placed.
    """
    self._input_dataset = input_dataset._apply_debug_options()  # pylint: disable=protected-access
    self._target_device = target_device
    spec = framework_device.DeviceSpec().from_string(self._target_device)
    self._is_gpu_target = (spec.device_type == "GPU")
    self._source_device_string = source_device
    self._source_device = ops.convert_to_tensor(source_device)

    wrap_ds_variant = gen_dataset_ops.wrap_dataset_variant(
        self._input_dataset._variant_tensor)  # pylint: disable=protected-access

    @def_function.function()
    def _init_func():
      # [FUNCTION DOC]
      # HELPER: traced function run on the source device. Unwraps the input dataset variant,
      # creates an anonymous iterator, initializes it with the dataset, and returns the
      # iterator's string handle (as a string tensor) once initialization has completed.
      """Creates an iterator for the input dataset.

      Returns:
        A `string` tensor that encapsulates the iterator created.
      """
      ds_variant = gen_dataset_ops.unwrap_dataset_variant(wrap_ds_variant)
      resource = gen_dataset_ops.anonymous_iterator(
          **self._input_dataset._flat_structure)  # pylint: disable=protected-access
      with ops.control_dependencies(
          [gen_dataset_ops.make_iterator(ds_variant, resource)]):
        return gen_dataset_ops.iterator_to_string_handle(resource)

    init_func_concrete = _init_func.get_concrete_function()  # pylint: disable=protected-access

    @def_function.function()
    def _remote_init_func():
      # [FUNCTION DOC]
      # HELPER: traced function that calls the init function on the source device through a
      # remote call and returns the resulting string handle as a one-element list of string
      # tensors.
      return functional_ops.remote_call(
          target=self._source_device,
          args=init_func_concrete.captured_inputs,
          Tout=[dtypes.string],
          f=init_func_concrete)

    self._init_func = _remote_init_func.get_concrete_function()  # pylint: disable=protected-access
    self._init_captured_args = self._init_func.captured_inputs

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
    def _next_func(string_handle):
      # [FUNCTION DOC]
      # HELPER: traced function run on the source device. Input: string_handle, a scalar string
      # tensor naming the iterator made by the init function. Rebuilds the iterator from the
      # handle (using this dataset's legacy types, shapes and classes), gets the next element and
      # returns it flattened to a list of tensors. Signals end of input via the iterator.
      """Calls get_next for created iterator.

      Args:
        string_handle: An iterator string handle created by _init_func
      Returns:
        The elements generated from `input_dataset`
      """
      with ops.device(self._source_device_string):
        iterator = iterator_ops.Iterator.from_string_handle(
            string_handle,
            dataset_ops.get_legacy_output_types(self),
            dataset_ops.get_legacy_output_shapes(self),
            dataset_ops.get_legacy_output_classes(self))
      return structure.to_tensor_list(self.element_spec, iterator.get_next())

    next_func_concrete = _next_func.get_concrete_function()  # pylint: disable=protected-access

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.string)],
        experimental_attributes={"experimental_ints_on_device": True})
    def _remote_next_func(string_handle):
      # [FUNCTION DOC]
      # HELPER: traced function (integers kept on device) that calls the next function on the
      # source device through a remote call. Input: string_handle, scalar string tensor. Returns
      # the element components with the input dataset's flat types.
      return functional_ops.remote_call(
          target=self._source_device,
          args=[string_handle] + next_func_concrete.captured_inputs,
          Tout=self._input_dataset._flat_types,  # pylint: disable=protected-access
          f=next_func_concrete)

    self._next_func = _remote_next_func.get_concrete_function()
    self._next_captured_args = self._next_func.captured_inputs

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
    def _finalize_func(string_handle):
      # [FUNCTION DOC]
      # HELPER: traced cleanup function run on the source device. Input: string_handle, scalar
      # string tensor naming the iterator. Recovers the iterator resource and destroys it
      # (ignoring lookup errors), returning a 64-bit integer constant zero after destruction.
      """Destroys the iterator resource created.

      Args:
        string_handle: An iterator string handle created by _init_func
      Returns:
        Tensor constant 0
      """
      iterator_resource = gen_dataset_ops.iterator_from_string_handle_v2(
          string_handle,
          **self._input_dataset._flat_structure)  # pylint: disable=protected-access
      with ops.control_dependencies([
          resource_variable_ops.destroy_resource_op(
              iterator_resource, ignore_lookup_error=True)]):
        return array_ops.constant(0, dtypes.int64)

    finalize_func_concrete = _finalize_func.get_concrete_function()  # pylint: disable=protected-access

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
    def _remote_finalize_func(string_handle):
      # [FUNCTION DOC]
      # HELPER: traced function that calls the finalize function on the source device through a
      # remote call. Input: string_handle, scalar string tensor. Returns a one-element list
      # holding a 64-bit integer tensor.
      return functional_ops.remote_call(
          target=self._source_device,
          args=[string_handle] + finalize_func_concrete.captured_inputs,
          Tout=[dtypes.int64],
          f=finalize_func_concrete)

    self._finalize_func = _remote_finalize_func.get_concrete_function(  # pylint: disable=protected-access
    )
    self._finalize_captured_args = self._finalize_func.captured_inputs

    g = ops.get_default_graph()
    self._init_func.add_to_graph(g)
    self._next_func.add_to_graph(g)
    self._finalize_func.add_to_graph(g)
    # pylint: enable=protected-scope

    with ops.device(self._target_device):
      variant_tensor = gen_dataset_ops.generator_dataset(
          self._init_captured_args,
          self._next_captured_args,
          self._finalize_captured_args,
          init_func=self._init_func,
          next_func=self._next_func,
          finalize_func=self._finalize_func,
          **self._input_dataset._flat_structure)  # pylint: disable=protected-access
    super(_CopyToDeviceDataset, self).__init__(input_dataset, variant_tensor)

  # The one_shot_iterator implementation needs a 0 arg _make_dataset function
  # that thereby captures all the inputs required to create the dataset. Since
  # there are strings that are inputs to the GeneratorDataset which can't be
  # placed on a GPU, this fails for the GPU case. Therefore, disabling it for
  # GPU
  def make_one_shot_iterator(self):
    # [FUNCTION DOC]
    # Override that refuses one-shot iterators when the target device is a GPU (the generator's
    # string inputs cannot be placed on a GPU); raises a value error recommending an
    # initializable iterator. Otherwise defers to the parent implementation and returns its
    # iterator.
    if self._is_gpu_target:
      raise ValueError(
          "`make_one_shot_iterator` is not compatible with GPU execution. "
          "Please use `Dataset.make_initializable_iterator()` instead."
      )
    else:
      return super(_CopyToDeviceDataset, self).make_one_shot_iterator()


class _MapOnGpuDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # HELPER: private dataset implementation behind map_on_gpu.
  # Unary dataset that applies a user function to every element via the experimental map dataset
  # operation, wrapping the function so integers stay on the device. Holds the input dataset,
  # the inter-op parallelism flag and the wrapped function; the element structure is the
  # function's output structure.
  """A `Dataset` that maps a function over elements in its using a GPU."""

  def __init__(self, input_dataset, map_func, use_inter_op_parallelism=True):
    # [FUNCTION DOC]
    # Inputs: input_dataset; map_func, function from an element to a new element structure;
    # use_inter_op_parallelism, whether the function may use inter-op parallelism (default
    # true). Wraps the function with integers-on-device enabled, creates the experimental map
    # dataset variant tensor and calls the parent constructor. Returns nothing.
    """See `Dataset.map()` for details."""
    self._input_dataset = input_dataset
    self._use_inter_op_parallelism = use_inter_op_parallelism

    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func,
        self._transformation_name(),
        dataset=input_dataset,
        defun_kwargs={"experimental_ints_on_device": True})
    variant_tensor = ged_ops.experimental_map_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        **self._flat_structure)
    super(_MapOnGpuDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    # [FUNCTION DOC]
    # HELPER: returns a list holding the wrapped map function for framework tracking.
    return [self._map_func]

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the output structure of the wrapped map function.
    return self._map_func.output_structure

  def _transformation_name(self):
    # [FUNCTION DOC]
    # HELPER: returns the display name of this transformation for naming wrapped functions.
    return "map_on_gpu()"


def map_on_gpu(map_func):
  # [FUNCTION DOC]
  # Highly experimental, not exported. Input: map_func, function from an element structure to
  # another. Returns a transformation function that maps it over a dataset on the GPU; intended
  # for use after copying to a GPU device. No validation here.
  """Maps `map_func` across the elements of this dataset.

  NOTE: This is a highly experimental version of `tf.data.Dataset.map` that runs
  `map_func` on GPU. It must be used after applying the
  `tf.data.experimental.copy_to_device` transformation with a GPU device
  argument.

  Args:
    map_func: A function mapping a nested structure of tensors (having shapes
      and types defined by `self.output_shapes` and `self.output_types`) to
      another nested structure of tensors.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    # [FUNCTION DOC]
    # HELPER: closure returned by map_on_gpu. Returns the GPU map dataset built from the given
    # dataset and the captured function.
    return _MapOnGpuDataset(dataset, map_func)

  return _apply_fn
