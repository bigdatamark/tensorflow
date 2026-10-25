# [MODULE DOC]
# Unit tests for the plugin asset helpers: singleton behavior per graph, rejection of unnamed
# plugins, plugin name collision detection, listing all assets and honoring an explicit graph.
# Defines small asset subclasses used as fixtures. Run as a test binary.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.summary import plugin_asset


class _UnnamedPluginAsset(plugin_asset.PluginAsset):
  # [CLASS DOC]
  # HELPER: fixture for the plugin asset tests.
  # Concrete plugin asset with an empty assets mapping and no plugin name, so obtaining it from a
  # graph must fail. Also serves as the base for the named fixtures below.
  """An example asset with a dummy serialize method provided, but no name."""

  def assets(self):
    # [FUNCTION DOC]
    # HELPER: satisfies the abstract method for fixtures. Takes no inputs and returns an empty
    # mapping of assets.
    return {}


class _ExamplePluginAsset(_UnnamedPluginAsset):
  # [CLASS DOC]
  # HELPER: fixture for the plugin asset tests.
  # Named plugin asset (plugin name equal to its class name) with no assets.
  """Simple example asset."""
  plugin_name = "_ExamplePluginAsset"


class _OtherExampleAsset(_UnnamedPluginAsset):
  # [CLASS DOC]
  # HELPER: fixture for the plugin asset tests.
  # Second named plugin asset with a different plugin name, used to check that distinct plugins
  # get distinct instances.
  """Simple example asset."""
  plugin_name = "_OtherExampleAsset"


class _ExamplePluginThatWillCauseCollision(_UnnamedPluginAsset):
  # [CLASS DOC]
  # HELPER: fixture for the plugin asset tests.
  # Plugin asset that deliberately reuses the plugin name of the example asset class to trigger
  # the name collision error.
  plugin_name = "_ExamplePluginAsset"


class PluginAssetTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the plugin asset module functions, using the fixture asset classes above and
  # the default graph (reset per test by the base test case) or explicitly created graphs.

  def testGetPluginAsset(self):
    # [FUNCTION DOC]
    # Test: fetching the example asset twice from the default graph returns the same instance of
    # the right type; fetching a differently named asset returns a different instance.
    epa = plugin_asset.get_plugin_asset(_ExamplePluginAsset)
    self.assertIsInstance(epa, _ExamplePluginAsset)
    epa2 = plugin_asset.get_plugin_asset(_ExamplePluginAsset)
    self.assertIs(epa, epa2)
    opa = plugin_asset.get_plugin_asset(_OtherExampleAsset)
    self.assertIsNot(epa, opa)

  def testUnnamedPluginFails(self):
    # [FUNCTION DOC]
    # Test: requesting an asset class that has no plugin name raises a value error.
    with self.assertRaises(ValueError):
      plugin_asset.get_plugin_asset(_UnnamedPluginAsset)

  def testPluginCollisionDetected(self):
    # [FUNCTION DOC]
    # Test: after registering the example asset, requesting a different class with the same
    # plugin name raises a value error.
    plugin_asset.get_plugin_asset(_ExamplePluginAsset)
    with self.assertRaises(ValueError):
      plugin_asset.get_plugin_asset(_ExamplePluginThatWillCauseCollision)

  def testGetAllPluginAssets(self):
    # [FUNCTION DOC]
    # Test: after registering two differently named assets, the list of all assets contains
    # exactly those two instances (order ignored).
    epa = plugin_asset.get_plugin_asset(_ExamplePluginAsset)
    opa = plugin_asset.get_plugin_asset(_OtherExampleAsset)
    self.assertItemsEqual(plugin_asset.get_all_plugin_assets(), [epa, opa])

  def testRespectsGraphArgument(self):
    # [FUNCTION DOC]
    # Test: registers the same asset class in two separate graphs and asserts each graph's asset
    # list holds the instance created for that graph, so the graph argument is honored.
    g1 = ops.Graph()
    g2 = ops.Graph()
    e1 = plugin_asset.get_plugin_asset(_ExamplePluginAsset, g1)
    e2 = plugin_asset.get_plugin_asset(_ExamplePluginAsset, g2)

    self.assertEqual(e1, plugin_asset.get_all_plugin_assets(g1)[0])
    self.assertEqual(e2, plugin_asset.get_all_plugin_assets(g2)[0])

if __name__ == "__main__":
  googletest.main()
