# [MODULE DOC]
# HELPER: shared test helper module for distribution strategy tests of Keras.
# Defines named optimizer factories (v1 training optimizers and Keras V2 optimizers, each
# wrapped as a named object so test names are readable) with fixed small learning rates,
# the lists of them (v1, v2 and both), and three functions that build test combinations
# crossing a fixed set of distribution strategies with those optimizer lists. The only
# import-time effect is creating these module-level objects; no optimizer is instantiated
# until a factory is called.
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
"""Strategy and optimizer combinations for combinations.combine()."""

from tensorflow.python.distribute import strategy_combinations as strategy_combinations_base
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_keras_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_keras_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_keras_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_keras_v2
from tensorflow.python.keras.optimizer_v2 import ftrl as ftrl_keras_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_keras_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_keras_v2
from tensorflow.python.training import adagrad
from tensorflow.python.training import adam
from tensorflow.python.training import ftrl
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import rmsprop


gradient_descent_optimizer_v1_fn = combinations.NamedObject(
    "GradientDescentV1",
    lambda: gradient_descent.GradientDescentOptimizer(0.001))
adagrad_optimizer_v1_fn = combinations.NamedObject(
    "AdagradV1", lambda: adagrad.AdagradOptimizer(0.001))
adam_optimizer_v1_fn = combinations.NamedObject(
    "AdamV1", lambda: adam.AdamOptimizer(0.001, epsilon=1))
ftrl_optimizer_v1_fn = combinations.NamedObject(
    "FtrlV1", lambda: ftrl.FtrlOptimizer(0.001))
rmsprop_optimizer_v1_fn = combinations.NamedObject(
    "RmsPropV1", lambda: rmsprop.RMSPropOptimizer(0.001))

# TODO(shiningsun): consider adding the other v1 optimizers
optimizers_v1 = [
    gradient_descent_optimizer_v1_fn, adagrad_optimizer_v1_fn,
    ftrl_optimizer_v1_fn, rmsprop_optimizer_v1_fn
]

adadelta_optimizer_keras_v2_fn = combinations.NamedObject(
    "AdadeltaKerasV2", lambda: adadelta_keras_v2.Adadelta(0.001))
adagrad_optimizer_keras_v2_fn = combinations.NamedObject(
    "AdagradKerasV2", lambda: adagrad_keras_v2.Adagrad(0.001))
adam_optimizer_keras_v2_fn = combinations.NamedObject(
    "AdamKerasV2", lambda: adam_keras_v2.Adam(0.001, epsilon=1.0))
adamax_optimizer_keras_v2_fn = combinations.NamedObject(
    "AdamaxKerasV2", lambda: adamax_keras_v2.Adamax(0.001, epsilon=1.0))
nadam_optimizer_keras_v2_fn = combinations.NamedObject(
    "NadamKerasV2", lambda: nadam_keras_v2.Nadam(0.001, epsilon=1.0))
ftrl_optimizer_keras_v2_fn = combinations.NamedObject(
    "FtrlKerasV2", lambda: ftrl_keras_v2.Ftrl(0.001))
gradient_descent_optimizer_keras_v2_fn = combinations.NamedObject(
    "GradientDescentKerasV2", lambda: gradient_descent_keras_v2.SGD(0.001))
rmsprop_optimizer_keras_v2_fn = combinations.NamedObject(
    "RmsPropKerasV2", lambda: rmsprop_keras_v2.RMSprop(0.001))

# TODO(shiningsun): consider adding the other v2 optimizers
optimizers_v2 = [
    gradient_descent_optimizer_keras_v2_fn, adagrad_optimizer_keras_v2_fn
]

optimizers_v1_and_v2 = optimizers_v1 + optimizers_v2


def distributions_and_v1_optimizers():
  # [FUNCTION DOC]
  # HELPER: parameter generator for distributed Keras tests.
  # Takes no arguments. Returns a test combination crossing four strategies (one device,
  # mirrored with GPU and CPU, mirrored with two GPUs, and the two-GPU variant without merge
  # call) with the list of v1 optimizer factories. No side effects.
  """A common set of combination with DistributionStrategies and Optimizers."""
  return combinations.combine(
      distribution=[
          strategy_combinations_base.one_device_strategy,
          strategy_combinations_base.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations_base.mirrored_strategy_with_two_gpus,
          strategy_combinations_base
          .mirrored_strategy_with_two_gpus_no_merge_call,
      ],
      optimizer_fn=optimizers_v1)


def distributions_and_v2_optimizers():
  # [FUNCTION DOC]
  # HELPER: parameter generator for distributed Keras tests.
  # Takes no arguments. Returns a test combination crossing the same four strategies as the
  # v1 variant with the list of Keras V2 optimizer factories. No side effects.
  """A common set of combination with DistributionStrategies and Optimizers."""
  return combinations.combine(
      distribution=[
          strategy_combinations_base.one_device_strategy,
          strategy_combinations_base.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations_base.mirrored_strategy_with_two_gpus,
          strategy_combinations_base
          .mirrored_strategy_with_two_gpus_no_merge_call,
      ],
      optimizer_fn=optimizers_v2)


def distributions_and_v1_and_v2_optimizers():
  # [FUNCTION DOC]
  # HELPER: parameter generator for distributed Keras tests.
  # Takes no arguments. Returns a test combination crossing the same four strategies with
  # the concatenated list of v1 and Keras V2 optimizer factories. No side effects.
  """A common set of combination with DistributionStrategies and Optimizers."""
  return combinations.combine(
      distribution=[
          strategy_combinations_base.one_device_strategy,
          strategy_combinations_base.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations_base.mirrored_strategy_with_two_gpus,
          strategy_combinations_base
          .mirrored_strategy_with_two_gpus_no_merge_call,
      ],
      optimizer_fn=optimizers_v1_and_v2)
