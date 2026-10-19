# [MODULE DOC]
# Data-specific fork of the nested structure utilities. Thin wrappers (is_nested, flatten,
# assert_same_structure, pack_sequence_as, map_structure, assert_shallow_structure,
# flatten_up_to, map_structure_up_to) that delegate to the shared nest utility module using its
# data mode, in which lists are not a level of nesting (they are treated as tensor-like leaves)
# and sparse tensor values are atomic. Used throughout the dataset code. No import-time side
# effects beyond imports.
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

"""## Functions for working with arbitrarily nested sequences of elements.

NOTE(mrry): This fork of the `tensorflow.python.util.nest` module
makes two changes:

1. It removes support for lists as a level of nesting in nested structures.
2. It adds support for `SparseTensorValue` as an atomic element.

The motivation for this change is twofold:

1. It seems more natural for lists to be treated (e.g. in Dataset constructors)
   as tensors, rather than lists of (lists of...) tensors.
2. This is needed because `SparseTensorValue` is implemented as a `namedtuple`
   that would normally be flattened and we want to be able to create sparse
   tensor from `SparseTensorValue's similarly to creating tensors from numpy
   arrays.
"""

from tensorflow.python.util import nest_util


def is_nested(structure):
  # [FUNCTION DOC]
  # Returns whether the given value is a nested structure under data mode rules of the shared
  # nest utilities (lists are not treated as nesting). Takes any value; no side effects.
  return nest_util.is_nested(nest_util.Modality.DATA, structure)


def flatten(structure):
  # [FUNCTION DOC]
  # Flattens a nested structure into a flat list of its leaves, using data mode of the shared
  # nest utilities (lists are leaves, sparse tensor values are atomic). Returns the list.
  return nest_util.flatten(nest_util.Modality.DATA, structure)


def assert_same_structure(nest1, nest2, check_types=True):
  # [FUNCTION DOC]
  # Checks that two nested structures are nested identically. check_types (default true) also
  # requires matching sequence types (for mappings, the keys are part of the type). Returns
  # nothing. Raises a value error on differing element counts or nesting, and a type error on
  # differing sequence types when checking types. Delegates to the shared nest utilities in
  # data mode.
  """Asserts that two structures are nested in the same way.

  Args:
    nest1: an arbitrarily nested structure.
    nest2: an arbitrarily nested structure.
    check_types: if `True` (default) types of sequences should be same as
      well. For dictionary, "type" of dictionary is considered to include its
      keys. In other words, two dictionaries with different keys are considered
      to have a different "type". If set to `False`, two iterables are
      considered same as long as they yield the elements that have same
      structures.

  Raises:
    ValueError: If the two structures do not have the same number of elements or
      if the two structures are not nested in the same way.
    TypeError: If the two structures differ in the type of sequence in any of
      their substructures. Only possible if `check_types` is `True`.
  """
  nest_util.assert_same_structure(
      nest_util.Modality.DATA, nest1, nest2, check_types
  )


def pack_sequence_as(structure, flat_sequence):
  # [FUNCTION DOC]
  # Packs a flat sequence into the nested shape of structure and returns it. If structure is a
  # scalar, flat_sequence must hold a single element, which is returned. Composite values are
  # not expanded. Raises a value error if the element counts differ. Delegates to the shared
  # nest utilities in data mode. No side effects.
  """Returns a given flattened sequence packed into a nest.

  If `structure` is a scalar, `flat_sequence` must be a single-element list;
  in this case the return value is `flat_sequence[0]`.

  Args:
    structure: tuple or list constructed of scalars and/or other tuples/lists,
      or a scalar.  Note: numpy arrays are considered scalars.
    flat_sequence: flat sequence to pack.

  Returns:
    packed: `flat_sequence` converted to have the same recursive structure as
      `structure`.

  Raises:
    ValueError: If nest and structure have different element counts.
  """
  return nest_util.pack_sequence_as(
      nest_util.Modality.DATA, structure, flat_sequence, expand_composites=False
  )


def map_structure(func, *structure, **check_types_dict):
  # [FUNCTION DOC]
  # Applies func to corresponding entries of one or more same-shaped structures and returns a
  # new structure with the results. func is a callable taking one argument per structure;
  # structure are the nested inputs; the only valid keyword is the check_types flag (default
  # true) controlling whether sequence types must match; when false, the first structure's types
  # are used. Raises a type error if func is not callable or structures mismatch in depth, and a
  # value error if no structure is given, types mismatch, or an unknown keyword is passed.
  # Delegates to the shared nest utilities in data mode.
  """Applies `func` to each entry in `structure` and returns a new structure.

  Applies `func(x[0], x[1], ...)` where x[i] is an entry in
  `structure[i]`.  All structures in `structure` must have the same arity,
  and the return value will contain the results in the same structure.

  Args:
    func: A callable that accepts as many arguments are there are structures.
    *structure: scalar, or tuple or list of constructed scalars and/or other
      tuples/lists, or scalars.  Note: numpy arrays are considered scalars.
    **check_types_dict: only valid keyword argument is `check_types`. If set to
      `True` (default) the types of iterables within the structures have to be
      same (e.g. `map_structure(func, [1], (1,))` raises a `TypeError`
      exception). To allow this set this argument to `False`.

  Returns:
    A new structure with the same arity as `structure`, whose values correspond
    to `func(x[0], x[1], ...)` where `x[i]` is a value in the corresponding
    location in `structure[i]`. If there are different sequence types and
    `check_types` is `False` the sequence types of the first structure will be
    used.

  Raises:
    TypeError: If `func` is not callable or if the structures do not match
      each other by depth tree.
    ValueError: If no structure is provided or if the structures do not match
      each other by type.
    ValueError: If wrong keyword arguments are provided.
  """
  return nest_util.map_structure(
      nest_util.Modality.DATA, func, *structure, **check_types_dict
  )


def assert_shallow_structure(shallow_tree, input_tree, check_types=True):
  # [FUNCTION DOC]
  # Checks that shallow_tree is a shallow version of input_tree (input_tree could be obtained by
  # replacing leaves of shallow_tree with deeper structures). check_types (default true)
  # requires matching sequence types. Returns nothing. Raises a type error if shallow_tree is a
  # sequence but input_tree is not or types differ (only when checking types), and a value error
  # on differing sequence lengths. Delegates to the shared nest utilities in data mode.
  """Asserts that `shallow_tree` is a shallow structure of `input_tree`.

  That is, this function tests if the `input_tree` structure can be created from
  the `shallow_tree` structure by replacing its leaf nodes with deeper
  tree structures.

  Examples:

  The following code will raise an exception:
  ```python
    shallow_tree = ["a", "b"]
    input_tree = ["c", ["d", "e"], "f"]
    assert_shallow_structure(shallow_tree, input_tree)
  ```

  The following code will not raise an exception:
  ```python
    shallow_tree = ["a", "b"]
    input_tree = ["c", ["d", "e"]]
    assert_shallow_structure(shallow_tree, input_tree)
  ```

  Args:
    shallow_tree: an arbitrarily nested structure.
    input_tree: an arbitrarily nested structure.
    check_types: if `True` (default) the sequence types of `shallow_tree` and
      `input_tree` have to be the same.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`. Only raised if `check_types` is `True`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
  nest_util.assert_shallow_structure(
      nest_util.Modality.DATA, shallow_tree, input_tree, check_types
  )


def flatten_up_to(shallow_tree, input_tree):
  # [FUNCTION DOC]
  # Flattens input_tree only as deep as shallow_tree and returns the resulting list; deeper
  # substructures remain as single elements. If both are non-sequences, returns a one-element
  # list holding input_tree. Raises a type error if shallow_tree is a sequence but input_tree is
  # not or their sequence types differ, and a value error if lengths differ. Delegates to the
  # shared nest utilities in data mode. No side effects.
  """Flattens `input_tree` up to `shallow_tree`.

  Any further depth in structure in `input_tree` is retained as elements in the
  partially flatten output.

  If `shallow_tree` and `input_tree` are not sequences, this returns a
  single-element list: `[input_tree]`.

  Use Case:

  Sometimes we may wish to partially flatten a nested sequence, retaining some
  of the nested structure. We achieve this by specifying a shallow structure,
  `shallow_tree`, we wish to flatten up to.

  The input, `input_tree`, can be thought of as having the same structure as
  `shallow_tree`, but with leaf nodes that are themselves tree structures.

  Examples:

  ```python
  input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
  shallow_tree = [[True, True], [False, True]]

  flattened_input_tree = flatten_up_to(shallow_tree, input_tree)
  flattened_shallow_tree = flatten_up_to(shallow_tree, shallow_tree)

  # Output is:
  # [[2, 2], [3, 3], [4, 9], [5, 5]]
  # [True, True, False, True]
  ```

  ```python
  input_tree = [[('a', 1), [('b', 2), [('c', 3), [('d', 4)]]]]]
  shallow_tree = [['level_1', ['level_2', ['level_3', ['level_4']]]]]

  input_tree_flattened_as_shallow_tree = flatten_up_to(shallow_tree, input_tree)
  input_tree_flattened = flatten(input_tree)

  # Output is:
  # [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
  # ['a', 1, 'b', 2, 'c', 3, 'd', 4]
  ```

  Non-Sequence Edge Cases:

  ```python
  flatten_up_to(0, 0)  # Output: [0]
  flatten_up_to(0, [0, 1, 2])  # Output: [[0, 1, 2]]
  flatten_up_to([0, 1, 2], 0)  # Output: TypeError
  flatten_up_to([0, 1, 2], [0, 1, 2])  # Output: [0, 1, 2]
  ```

  Args:
    shallow_tree: a possibly pruned structure of input_tree.
    input_tree: an arbitrarily nested structure or a scalar object.
      Note, numpy arrays are considered scalars.

  Returns:
    A Python list, the partially flattened version of `input_tree` according to
    the structure of `shallow_tree`.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
  return nest_util.flatten_up_to(
      nest_util.Modality.DATA, shallow_tree, input_tree
  )


def map_structure_up_to(shallow_tree, func, *inputs):
  # [FUNCTION DOC]
  # Applies func to the partially flattened inputs. Each input is flattened only down to the
  # structure of shallow_tree; func receives one corresponding element from each input (so it
  # must accept as many arguments as there are inputs). Returns a structure shaped like
  # shallow_tree holding the results. Delegates to the shared nest utilities in data mode (lists
  # are not treated as nested). Raises type errors if the shallow tree is a sequence but an input
  # is not, or sequence types differ, and a value error if sequence lengths differ.
  """Applies a function or op to a number of partially flattened inputs.

  The `inputs` are flattened up to `shallow_tree` before being mapped.

  Use Case:

  Sometimes we wish to apply a function to a partially flattened
  sequence (for example when the function itself takes sequence inputs). We
  achieve this by specifying a shallow structure, `shallow_tree` we wish to
  flatten up to.

  The `inputs`, can be thought of as having the same structure as
  `shallow_tree`, but with leaf nodes that are themselves tree structures.

  This function, therefore, will return something with the same base structure
  as `shallow_tree`.

  Examples:

  ```python
  ab_tuple = collections.namedtuple("ab_tuple", "a, b")
  op_tuple = collections.namedtuple("op_tuple", "add, mul")
  inp_val = ab_tuple(a=2, b=3)
  inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
  out = map_structure_up_to(inp_val, lambda val, ops: (val + ops.add) * ops.mul,
                            inp_val, inp_ops)

  # Output is: ab_tuple(a=6, b=15)
  ```

  ```python
  data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]
  name_list = ['evens', ['odds', 'primes']]
  out = map_structure_up_to(
      name_list,
      lambda name, sec: "first_{}_{}".format(len(sec), name),
      name_list, data_list)

  # Output is: ['first_4_evens', ['first_5_odds', 'first_3_primes']]
  ```

  Args:
    shallow_tree: a shallow tree, common to all the inputs.
    func: callable which will be applied to each input individually.
    *inputs: arbitrarily nested combination of objects that are compatible with
        shallow_tree. The function `func` is applied to corresponding
        partially flattened elements of each input, so the function must support
        arity of `len(inputs)`.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.

  Returns:
    result of repeatedly applying `func`, with same structure as
    `shallow_tree`.
  """
  return nest_util.map_structure_up_to(
      nest_util.Modality.DATA, shallow_tree, func, *inputs
  )
