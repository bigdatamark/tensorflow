# [PACKAGE DOC]
# Package of static (non-executing) analyses over Python ASTs used by autograph: symbol activity
# analysis, liveness, and reaching definitions (the latter two being dataflow analyses on the
# control-flow graph that use the activity information), plus related annotation utilities. This
# file only carries the package docstring; it exports nothing and has no import-time side effects.
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
"""Static information resolution.

This module contains utilities to help annotate AST nodes with as much runtime
information as can be possibly extracted without actually executing the code,
under that assumption that the context in which the code will run is known.

Overall, the different analyses have the functions listed below:

 * activity: inventories symbols read, written to, params, etc. at different
     levels
 * liveness, reaching_definitions: dataflow analyses based on the program's CFG
     and using the symbol information gathered by activity analysis
"""

