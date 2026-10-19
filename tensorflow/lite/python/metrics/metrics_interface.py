# [MODULE DOC]
# Defines the abstract interface for the TFLite metrics helper: a set of counter increment, converter
# parameter, converter error and converter latency reporting hooks. Concrete implementations live in
# the portable (no-op) and non-portable (monitoring based) metrics modules. Helper module.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Python TFLite metrics helper interface."""
import abc


class TFLiteMetricsInterface(metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract base class (abstract-base-class metaclass) listing the metric reporting operations that
  # every TFLite metrics implementation must provide. Holds no state. Subclasses: the portable
  # no-op metrics and the non-portable monitoring based metrics.
  """Abstract class for TFLiteMetrics."""

  @abc.abstractmethod
  def increase_counter_debugger_creation(self):
    # [FUNCTION DOC]
    # Abstract: count one creation of a quantization debugger. Base version raises not-implemented.
    raise NotImplementedError

  @abc.abstractmethod
  def increase_counter_interpreter_creation(self):
    # [FUNCTION DOC]
    # Abstract: count one creation of an interpreter. Base version raises not-implemented.
    raise NotImplementedError

  @abc.abstractmethod
  def increase_counter_converter_attempt(self):
    # [FUNCTION DOC]
    # Abstract: count one conversion attempt. Base version raises not-implemented.
    raise NotImplementedError

  @abc.abstractmethod
  def increase_counter_converter_success(self):
    # [FUNCTION DOC]
    # Abstract: count one successful conversion. Base version raises not-implemented.
    raise NotImplementedError

  @abc.abstractmethod
  def set_converter_param(self, name, value):
    # [FUNCTION DOC]
    # Abstract: record a converter parameter; name is the parameter name, value its text value.
    # Base version raises not-implemented.
    raise NotImplementedError

  @abc.abstractmethod
  def set_converter_error(self, error_data):
    # [FUNCTION DOC]
    # Abstract: record a converter error described by an error data message. Base version raises
    # not-implemented.
    raise NotImplementedError

  @abc.abstractmethod
  def set_converter_latency(self, value):
    # [FUNCTION DOC]
    # Abstract: record the conversion latency in milliseconds (integer). Base version raises
    # not-implemented.
    raise NotImplementedError
