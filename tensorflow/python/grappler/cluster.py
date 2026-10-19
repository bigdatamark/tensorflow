# [MODULE DOC]
# Python interface for Grappler clusters (the hardware description and measurement environment
# used by the graph optimizer). Exposes a Cluster class wrapping a native cluster handle (real
# local cluster or virtual cluster built from device descriptions) and a context manager that
# creates a cluster and shuts it down afterwards. Imports native bindings and protocol buffer
# types; no other import-time side effects.
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
"""A python interface for Grappler clusters."""

import contextlib

from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.grappler import _pywrap_tf_cluster as tf_cluster


class Cluster(object):
  # [CLASS DOC]
  # Wrapper around a native Grappler cluster handle. Holds the native cluster (or nothing after
  # shutdown) and whether a timeline should be generated when measuring costs. Created with either
  # a local cluster or a virtual cluster from a list of device descriptions. The native cluster is
  # shut down explicitly or when the object is garbage collected.
  """Grappler Clusters."""

  def __init__(self,
               allow_soft_placement=True,
               disable_detailed_stats=True,
               disable_timeline=True,
               devices=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: allow_soft_placement, whether illegal placements are fixed instead of
    # failing (default true); disable_detailed_stats, whether detailed statistics are off (default
    # true); disable_timeline, whether timeline information is off (default true, stored inverted
    # for later use); devices, optional list of named device descriptions (default none).
    # With no devices, creates a native local cluster using the first two options; with devices,
    # serializes each one and creates a native virtual cluster (the first two options and the
    # timeline flag are then not passed to it). Returns nothing.
    # KNOWN ISSUE: for virtual clusters the allow_soft_placement and disable_detailed_stats
    # arguments are silently ignored.
    """Creates a Cluster.

    Args:
      allow_soft_placement: If True, TF will automatically fix illegal
        placements instead of erroring out if the placement isn't legal.
      disable_detailed_stats: If True, detailed statistics will not be
        available.
      disable_timeline: If True, the timeline information will not be reported.
      devices: A list of devices of type device_properties_pb2.NamedDevice.
        If None, a device list will be created based on the spec of
        the local machine.
    """
    self._tf_cluster = None
    self._generate_timeline = not disable_timeline

    if devices is None:
      self._tf_cluster = tf_cluster.TF_NewCluster(allow_soft_placement,
                                                  disable_detailed_stats)
    else:
      devices_serialized = [device.SerializeToString() for device in devices]
      self._tf_cluster = tf_cluster.TF_NewVirtualCluster(devices_serialized)

  def Shutdown(self):
    # [FUNCTION DOC]
    # Shuts down the native cluster if it exists and clears the handle. Safe to call repeatedly
    # (later calls do nothing). Takes no inputs, returns nothing.
    if self._tf_cluster is not None:
      tf_cluster.TF_ShutdownCluster(self._tf_cluster)
      self._tf_cluster = None

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer: shuts the native cluster down when the object is garbage collected. Returns nothing.
    self.Shutdown()

  @property
  def tf_cluster(self):
    # [FUNCTION DOC]
    # Property returning the underlying native cluster handle, or nothing after shutdown.
    return self._tf_cluster

  def ListDevices(self):
    # [FUNCTION DOC]
    # Lists the devices of this cluster. Returns a list of named device description messages parsed
    # from the native serialized form, or an empty list if the cluster has been shut down.
    """Returns a list of available hardware devices."""
    if self._tf_cluster is None:
      return []
    return [device_properties_pb2.NamedDevice.FromString(device)
            for device in tf_cluster.TF_ListDevices(self._tf_cluster)]

  def ListAvailableOps(self):
    # [FUNCTION DOC]
    # Returns the alphabetically sorted list of all operation names known to the native runtime.
    # Does not depend on the cluster state (works even after shutdown).
    """Returns a list of all available operations (sorted alphabetically)."""
    return tf_cluster.TF_ListAvailableOps()

  def GetSupportedDevices(self, item):
    # [FUNCTION DOC]
    # Input: item, a Grappler item wrapper exposing its native handle. Returns what the native call
    # returns: the supported devices for the nodes of the item on this cluster. No docstring
    # exists. Passing a shut-down cluster hands a null handle to the native call.
    return tf_cluster.TF_GetSupportedDevices(self._tf_cluster, item.tf_item)

  def EstimatePerformance(self, device):
    # [FUNCTION DOC]
    # Input: device, a named device description message. Serializes it and asks the native layer to
    # estimate its performance. Returns the native result. Does not use the cluster itself.
    return tf_cluster.TF_EstimatePerformance(device.SerializeToString())

  def MeasureCosts(self, item):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring calls the result a triplet of op_perfs, runtime, step_stats, which
    # matches the code only loosely: the second value is the native run time and the third is a
    # parsed step statistics message; the code returns a plain tuple of those three.
    # Measures the cost of running a Grappler item on this cluster. Input: item, a Grappler item
    # wrapper. Runs the native measurement (generating a timeline if enabled at construction) and
    # parses results. Returns a tuple of: list of per-operation performance messages, the run time,
    # and a step statistics message. Side effect: runs the item on the cluster.
    """Returns the cost of running the specified item.

    Args:
      item: The item for which to measure the costs.
    Returns: The triplet op_perfs, runtime, step_stats.
    """
    op_perf_bytes_list, run_time, step_stats_bytes = tf_cluster.TF_MeasureCosts(
        item.tf_item, self._tf_cluster, self._generate_timeline)

    op_perfs = [op_performance_data_pb2.OpPerformance.FromString(op_perf_bytes)
                for op_perf_bytes in op_perf_bytes_list]
    return (op_perfs, run_time,
            step_stats_pb2.StepStats.FromString(step_stats_bytes))

  def DeterminePeakMemoryUsage(self, item):
    # [FUNCTION DOC]
    # Input: item, a Grappler item wrapper. Asks the native layer for the peak memory usage when
    # running the item on this cluster. Returns the native result: a mapping indexed by device name.
    """Returns a snapshot of the peak memory usage.

    Args:
      item: The item for which to measure the costs.
    Returns: A hashtable indexed by device name.
    """
    return tf_cluster.TF_DeterminePeakMemoryUsage(item.tf_item,
                                                  self._tf_cluster)


@contextlib.contextmanager
def Provision(allow_soft_placement=True,
              disable_detailed_stats=True,
              disable_timeline=True,
              devices=None):
  # [FUNCTION DOC]
  # Context manager that creates a Cluster and shuts it down afterwards. Inputs (all optional) are
  # passed to the Cluster constructor: allow_soft_placement (default true), disable_detailed_stats
  # (default true), disable_timeline (default true), devices (default none). Yields the cluster.
  # KNOWN ISSUE: the shutdown is not in a finally clause, so if the with-body raises, the cluster
  # is not shut down until it is garbage collected.
  cluster = Cluster(allow_soft_placement, disable_detailed_stats,
                    disable_timeline, devices)
  yield cluster
  cluster.Shutdown()
