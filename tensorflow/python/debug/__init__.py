# [PACKAGE DOC]
# Public Python API of the TensorFlow Debugger (tfdbg). Re-exports, from the debug library and
# wrappers subpackages, the dump data classes and loaders (debug dump directory, tensor datum,
# tensor loading and NaN/Inf check), gradient debugging, graph reconstruction, debug watch helpers,
# the watch options class, and the debug wrapper sessions and session hooks (dumping, gRPC,
# TensorBoard, local command-line). Import-time side effect: calls the utility that removes every
# public name not listed in this module's documentation, so only the documented symbols remain.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Public Python API of TensorFlow Debugger (tfdbg).

See the [TFDBG](https://www.tensorflow.org/guide/debugger) guide.

@@add_debug_tensor_watch
@@watch_graph
@@watch_graph_with_denylists
@@DebugTensorDatum
@@DebugDumpDir
@@load_tensor_from_event
@@load_tensor_from_event_file
@@has_inf_or_nan
@@DumpingDebugHook
@@DumpingDebugWrapperSession
@@GrpcDebugHook
@@GrpcDebugWrapperSession
@@LocalCLIDebugHook
@@LocalCLIDebugWrapperSession
@@TensorBoardDebugHook
@@TensorBoardDebugWrapperSession
@@WatchOptions

@@reconstruct_non_debug_graph_def

@@GradientsDebugger
@@clear_gradient_debuggers
"""

# pylint: disable=unused-imports
from tensorflow.python.debug.lib.debug_data import DebugDumpDir
from tensorflow.python.debug.lib.debug_data import DebugTensorDatum
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
from tensorflow.python.debug.lib.debug_data import load_tensor_from_event
from tensorflow.python.debug.lib.debug_data import load_tensor_from_event_file

from tensorflow.python.debug.lib.debug_gradients import GradientsDebugger

from tensorflow.python.debug.lib.debug_graphs import reconstruct_non_debug_graph_def

from tensorflow.python.debug.lib.debug_utils import add_debug_tensor_watch
from tensorflow.python.debug.lib.debug_utils import watch_graph
from tensorflow.python.debug.lib.debug_utils import watch_graph_with_denylists

from tensorflow.python.debug.wrappers.dumping_wrapper import DumpingDebugWrapperSession
from tensorflow.python.debug.wrappers.framework import WatchOptions
from tensorflow.python.debug.wrappers.grpc_wrapper import GrpcDebugWrapperSession
from tensorflow.python.debug.wrappers.grpc_wrapper import TensorBoardDebugWrapperSession
from tensorflow.python.debug.wrappers.hooks import DumpingDebugHook
from tensorflow.python.debug.wrappers.hooks import GrpcDebugHook
from tensorflow.python.debug.wrappers.hooks import LocalCLIDebugHook
from tensorflow.python.debug.wrappers.hooks import TensorBoardDebugHook
from tensorflow.python.debug.wrappers.local_cli_wrapper import LocalCLIDebugWrapperSession

from tensorflow.python.util import all_util as _all_util


_all_util.remove_undocumented(__name__)
