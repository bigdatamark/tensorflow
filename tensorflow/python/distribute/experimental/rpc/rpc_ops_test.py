# [MODULE DOC]
# Unit tests for the experimental RPC ops module (rpc_ops). They start gRPC servers on unused
# local ports, register TensorFlow functions on them, and call them through clients, covering the
# raw generated ops, the Server/Client factories, the gRPC server and client classes, output and
# input specs, timeouts, registration ordering, resource deletion, errors, captured variables and
# resources (queues, multi-device variables). Runs eagerly when executed as a script.
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
"""Tests for rpc_ops.py."""

import threading
import time
import numpy as np
import portpicker

from tensorflow.python.distribute.experimental.rpc import rpc_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function as eager_def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import nest


@test_util.with_eager_op_as_function
class RpcOpsTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the RPC ops module: raw ops, server/client wrappers, specs, timeouts,
  # error handling and resource lifetime. Shared fixture: two logical CPU devices are configured
  # in setUp. Each test picks an unused port and builds its own server and client.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the base setUp, then splits the first physical CPU into two logical CPU
    # devices so that tests can place resources on separate devices. Takes and returns nothing.
    super(RpcOpsTest, self).setUp()
    cpus = config.list_physical_devices("CPU")
    # Set 2 virtual CPUs
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])

  def test_generated_rpc_ops(self):
    # [FUNCTION DOC]
    # Scenario: use the low-level generated RPC ops directly. Defines a function multiplying two
    # integers, creates a server resource, registers the function under the name multiply, starts
    # the server, creates a client handle and issues a call with two variables (2 and 3).
    # Asserts the status code is zero and the returned value is 6, then deletes the server,
    # client and future resources.
    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def remote_fn(a, b):
      # [FUNCTION DOC]
      # Remote function under test: returns the product of two scalar integers.
      return math_ops.multiply(a, b)

    concrete_remote_fn = remote_fn.get_concrete_function()

    a = variables.Variable(2, dtype=dtypes.int32)
    b = variables.Variable(3, dtype=dtypes.int32)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.gen_rpc_ops.rpc_server(server_address=address)

    rpc_ops.gen_rpc_ops.rpc_server_register(
        server_resource,
        f=concrete_remote_fn,
        captured_inputs=concrete_remote_fn.captured_inputs,
        output_specs=rpc_ops.get_output_specs_from_function(concrete_remote_fn),
        method_name="multiply")

    rpc_ops.gen_rpc_ops.rpc_server_start(server_resource)
    client_handle, _ = rpc_ops.gen_rpc_ops.rpc_client(
        server_address=address, timeout_in_ms=5000)
    future_resource, deleter = rpc_ops.gen_rpc_ops.rpc_call(
        client_handle, args=[a, b], method_name="multiply", timeout_in_ms=0)

    error_code, _ = rpc_ops.gen_rpc_ops.rpc_check_status(future_resource)
    self.assertAllEqual(error_code, 0)
    self.assertAllEqual(
        rpc_ops.gen_rpc_ops.rpc_get_value(future_resource, Tout=[dtypes.int32]),
        [6])

    resource_variable_ops.EagerResourceDeleter(
        handle=server_resource, handle_device=server_resource.device)

    resource_variable_ops.EagerResourceDeleter(
        handle=client_handle, handle_device=client_handle.device)

    rpc_ops.gen_rpc_ops.delete_rpc_future_resource(future_resource, deleter)

  def test_exported_rpc_api_static_factory(self):
    # [FUNCTION DOC]
    # Scenario: create the server and clients through the static factory methods using the gRPC
    # protocol. Registers a multiply function, starts the server, and calls it with variables 2
    # and 3 via an explicit call with output specs, with a client created without a name, and via
    # the convenience method without output specs. Asserts every call is ok and returns 6, and
    # that the convenience method's documentation string names the method and server address.
    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def _remote_fn(a, b):
      # [FUNCTION DOC]
      # HELPER: remote function registered by the enclosing test.
      # Returns the product of two scalar integers.
      return math_ops.multiply(a, b)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.Server.create("grpc", address)
    server_resource.register("multiply", _remote_fn)

    server_resource.start()
    client = rpc_ops.Client.create("grpc", address=address, name="test_client")

    a = variables.Variable(2, dtype=dtypes.int32)
    b = variables.Variable(3, dtype=dtypes.int32)

    mul_or = client.call(
        args=[a, b],
        method_name="multiply",
        output_specs=tensor_spec.TensorSpec((), dtypes.int32))

    self.assertAllEqual(mul_or.is_ok(), True)
    self.assertAllEqual(mul_or.get_value(), 6)

    # Test empty client name
    client1 = rpc_ops.Client.create("grpc", address)
    mul_or = client1.call(
        args=[a, b],
        method_name="multiply",
        output_specs=tensor_spec.TensorSpec((), dtypes.int32))
    self.assertAllEqual(mul_or.is_ok(), True)
    self.assertAllEqual(mul_or.get_value(), 6)

    # Test without output_spec
    mul_or = client1.multiply(a, b)
    self.assertAllEqual(mul_or.is_ok(), True)
    self.assertAllEqual(mul_or.get_value(), 6)

    self.assertEqual(client1.multiply.__doc__,
                     "RPC Call for multiply method to server " + address)

  def test_rpc_ops_call_method(self):
    # [FUNCTION DOC]
    # Scenario: use the gRPC server and client classes with the generic call method. Registers a
    # multiply function (as a function object) and an add function (as a concrete function),
    # starts the server, and calls both by name with variables 2 and 3, passing output specs.
    # Also creates a second unnamed client that lists registered methods. Asserts all calls are
    # ok and return 6 (multiply) and 5 (add).

    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def _remote_fn(a, b):
      # [FUNCTION DOC]
      # HELPER: remote function registered by the enclosing test.
      # Returns the product of two scalar integers.
      return math_ops.multiply(a, b)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.GrpcServer(address)

    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def add_fn(a, b):
      # [FUNCTION DOC]
      # HELPER: remote function registered by the enclosing test.
      # Returns the sum of two scalar integers.
      return math_ops.add(a, b)

    # Register TF function
    server_resource.register("multiply", _remote_fn)

    # Register concrete Function
    server_resource.register("add", add_fn.get_concrete_function())

    server_resource.start()
    client = rpc_ops.GrpcClient(address=address, name="test_client")

    a = variables.Variable(2, dtype=dtypes.int32)
    b = variables.Variable(3, dtype=dtypes.int32)

    mul_or = client.call(
        args=[a, b],
        method_name="multiply",
        output_specs=tensor_spec.TensorSpec((), dtypes.int32))

    self.assertAllEqual(mul_or.is_ok(), True)
    self.assertAllEqual(mul_or.get_value(), 6)

    add_or = client.call(
        args=[a, b],
        method_name="add",
        output_specs=tensor_spec.TensorSpec((), dtypes.int32))

    self.assertAllEqual(add_or.is_ok(), True)
    self.assertAllEqual(add_or.get_value(), 5)

    # Test empty client name
    client1 = rpc_ops.GrpcClient(address, list_registered_methods=True)
    mul_or = client1.call(
        args=[a, b],
        method_name="multiply",
        output_specs=tensor_spec.TensorSpec((), dtypes.int32))
    self.assertAllEqual(mul_or.is_ok(), True)
    self.assertAllEqual(mul_or.get_value(), 6)

  def test_rpc_ops_non_blocking_convenience_methods(self):
    # [FUNCTION DOC]
    # Scenario: a client created with the option to list registered methods gets a per-method
    # convenience attribute. Registers a multiply function, starts the server, and calls the
    # attribute with variables 2 and 3. Asserts the returned future is ok with value 6 and that
    # the attribute's documentation string names the method and server address.
    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def _remote_fn(a, b):
      # [FUNCTION DOC]
      # HELPER: remote function registered by the enclosing test.
      # Returns the product of two scalar integers.
      return math_ops.multiply(a, b)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.GrpcServer(address)

    # Register TF function
    server_resource.register("multiply", _remote_fn)

    server_resource.start()
    a = variables.Variable(2, dtype=dtypes.int32)
    b = variables.Variable(3, dtype=dtypes.int32)

    client = rpc_ops.GrpcClient(address, list_registered_methods=True)

    mul_or = client.multiply(a, b)
    self.assertAllEqual(mul_or.is_ok(), True)
    self.assertAllEqual(mul_or.get_value(), 6)

    self.assertEqual(client.multiply.__doc__,
                     "RPC Call for multiply method to server " + address)

  def test_rpc_ops_blocking_convenience_methods(self):
    # [FUNCTION DOC]
    # Scenario: the blocking variant of the per-method convenience attribute. Registers a
    # multiply function, starts the server, creates a client that lists registered methods and
    # calls the blocking attribute with variables 2 and 3. Asserts the direct result is 6 and
    # that the attribute's documentation string names the method and server address.
    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def _remote_fn(a, b):
      # [FUNCTION DOC]
      # HELPER: remote function registered by the enclosing test.
      # Returns the product of two scalar integers.
      return math_ops.multiply(a, b)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.GrpcServer(address)

    # Register TF function
    server_resource.register("multiply", _remote_fn)

    server_resource.start()

    client = rpc_ops.GrpcClient(address, list_registered_methods=True)

    a = variables.Variable(2, dtype=dtypes.int32)
    b = variables.Variable(3, dtype=dtypes.int32)
    self.assertAllEqual(client.multiply_blocking(a, b), 6)

    self.assertEqual(
        client.multiply_blocking.__doc__,
        "RPC Call for multiply method to server " + address)

  def test_output_specs(self):
    # [FUNCTION DOC]
    # Scenario: output structures are inferred from the registered functions. Registers four
    # functions returning a dictionary, a boolean, an empty list and a nested structure, starts
    # the server and calls each through a convenience attribute with variable 2. Asserts every
    # call is ok and the values match the expected structures (dictionary with 2, true, empty
    # list, nested dictionary of tuples and lists of 2).

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def test_dict(val):
      # [FUNCTION DOC]
      # Remote function returning a dictionary with a single entry holding the input value.
      return {"key": val}

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def is_positive(a):
      # [FUNCTION DOC]
      # Remote function returning true when the integer input is greater than zero, else false.
      if a > 0:
        return True
      return False

    @eager_def_function.function(input_signature=[])
    def do_nothing():
      # [FUNCTION DOC]
      # Remote function with no inputs that returns an empty list.
      return []

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def test_nested_structure(v):
      # [FUNCTION DOC]
      # Remote function returning a nested structure: a dictionary whose entries are a tuple
      # containing the input and a list of two copies, and a one-element tuple of the input.
      return {"test": (v, [v, v]), "test1": (v,)}

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.GrpcServer(address)

    server_resource.register("test_dict", test_dict)
    server_resource.register("is_positive", is_positive)
    server_resource.register("test_nested_structure", test_nested_structure)
    server_resource.register("do_nothing", do_nothing)

    server_resource.start()

    client = rpc_ops.GrpcClient(
        address=address, name="test_client", list_registered_methods=True)

    a = variables.Variable(2, dtype=dtypes.int32)

    result_or = client.test_dict(a)
    self.assertAllEqual(result_or.is_ok(), True)
    nest.map_structure(self.assertAllEqual, result_or.get_value(), {"key": 2})

    result_or = client.is_positive(a)
    self.assertTrue(result_or.is_ok())
    self.assertTrue(result_or.get_value())

    result_or = client.test_nested_structure(a)
    self.assertAllEqual(result_or.is_ok(), True)
    nest.map_structure(self.assertAllEqual, result_or.get_value(), {
        "test": (2, [2, 2]),
        "test1": (2,)
    })

    result_or = client.do_nothing()
    self.assertAllEqual(result_or.is_ok(), True)
    self.assertAllEqual(result_or.get_value(), [])

  def test_input_specs(self):
    # [FUNCTION DOC]
    # Scenario: a registered function takes a dictionary input. Registers it, starts the server
    # and calls it with a dictionary of variables 2 and 3. Asserts the call is ok with value 5,
    # and that passing a plain list instead of a dictionary raises a type error.

    @eager_def_function.function(input_signature=[{
        "a": tensor_spec.TensorSpec([], dtypes.int32),
        "b": tensor_spec.TensorSpec([], dtypes.int32)
    }])
    def test_input_dict(value):
      # [FUNCTION DOC]
      # Remote function taking a dictionary with entries a and b; returns their sum.
      return math_ops.add(value["a"], value["b"])

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.GrpcServer(address)

    server_resource.register("test_input_dict", test_input_dict)

    server_resource.start()

    client = rpc_ops.GrpcClient(
        address=address, name="test_client", list_registered_methods=True)
    a = variables.Variable(2, dtype=dtypes.int32)
    b = variables.Variable(3, dtype=dtypes.int32)
    result_or = client.test_input_dict({"a": a, "b": b})
    self.assertAllEqual(result_or.is_ok(), True)
    self.assertAllEqual(result_or.get_value(), 5)

    with self.assertRaises(TypeError):
      client.test_input_dict([a, b])

  def test_call_register_ordering(self):
    # [FUNCTION DOC]
    # Scenario: ordering of client creation, method registration and server start. A plain
    # client can be created before the server exists, but a client that lists registered methods
    # with a tiny timeout raises a deadline-exceeded error. A background thread then registers
    # an add-assign function and starts the server after a one second delay, while a client
    # that lists methods is created and waits for it. Calls through both listed clients and the
    # plain client must succeed and the shared variable must end at 6. Registering another
    # method after the server started must raise a failed-precondition error.
    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)

    # Create client succeeds before server start and registration
    client = rpc_ops.GrpcClient(address)

    # Create client with list_registered_methods fails before server is started.
    with self.assertRaises(errors.DeadlineExceededError):
      rpc_ops.GrpcClient(
          address,
          name="client1",
          list_registered_methods=True,
          timeout_in_ms=1)

    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign_add(a):
      # [FUNCTION DOC]
      # Remote function adding the input to the captured variable; returns nothing.
      v.assign_add(a)

    @eager_def_function.function(input_signature=[])
    def read_var():
      # [FUNCTION DOC]
      # Remote function returning the current value of the captured variable.
      return v.value()

    server = rpc_ops.GrpcServer(address)

    def start_server():
      # [FUNCTION DOC]
      # HELPER: thread body for the enclosing test.
      # Waits one second, registers the add-assign method on the server, then starts it, so that
      # client creation has to wait for the server. Takes and returns nothing.
      # Delay server start to test whether client creation also waits
      # till server is up.
      time.sleep(1)
      server.register("assign_add", assign_add)
      server.start()

    t = threading.Thread(target=start_server)
    t.start()

    # Create same "client1" again should succeed.
    client1_with_listed_methods = rpc_ops.GrpcClient(
        address, name="client1", list_registered_methods=True)

    result_or = client1_with_listed_methods.assign_add(
        variables.Variable(2, dtype=dtypes.int64))
    self.assertAllEqual(result_or.is_ok(), True)

    result_or = client.call("assign_add",
                            [variables.Variable(2, dtype=dtypes.int64)])
    self.assertAllEqual(result_or.is_ok(), True)

    # Create client with registered methods
    client2_with_listed_methods = rpc_ops.GrpcClient(
        address=address, name="client2", list_registered_methods=True)

    result_or = client2_with_listed_methods.assign_add(
        variables.Variable(2, dtype=dtypes.int64))
    self.assertAllEqual(result_or.is_ok(), True)

    self.assertAllEqual(v, 6)

    # Register new method after server started.
    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        "All methods must be registered before starting the server"):
      server.register("read_var", read_var)

  def test_client_timeout(self):
    # [FUNCTION DOC]
    # Scenario: client and per-call timeouts. A thread starts the server after a one second
    # delay. Creating a client that lists methods with a 1 ms timeout raises a deadline-exceeded
    # error; a client without listing succeeds. After the server is ready, an explicit call with
    # a 1 ms timeout must fail with the deadline-exceeded code, while one with a 5000 ms timeout
    # succeeds. Finally the server is recreated and restarted with delay, and a listing client
    # calls the add convenience method with a generous timeout, which must succeed.
    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)

    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def add(a, b):
      # [FUNCTION DOC]
      # Remote function returning the sum of two scalar integers.
      return math_ops.add(a, b)

    server = rpc_ops.GrpcServer(address)

    def start_server():
      # [FUNCTION DOC]
      # HELPER: thread body for the enclosing test.
      # Waits one second, registers the add method and starts the server, simulating a late
      # server. Takes and returns nothing.
      # Delay server start to simulate deadline exceeded for 1st RPC call
      # response. Client waits till server is started, thus it can trigger
      # deadline exceeded.
      time.sleep(1)
      server.register("add", add)
      server.start()

    t = threading.Thread(target=start_server)
    t.start()

    def ensure_server_is_ready(client):
      # [FUNCTION DOC]
      # HELPER: polling helper for the enclosing test.
      # Repeatedly calls the add method through the given client with two constants until a call
      # succeeds or fails with an error other than unavailable, i.e. until the server answers.
      # Busy-loops without delay and returns nothing.
      server_ready = False
      while not server_ready:
        result_or = client.call(
            "add", [constant_op.constant(20),
                    constant_op.constant(30)])
        if result_or.is_ok():
          server_ready = True
        else:
          error_code, _ = result_or.get_error()
          if error_code == errors.UNAVAILABLE:
            server_ready = False
          else:
            server_ready = True
      return

    # Create client with list_registered_methods fails before server is started.
    with self.assertRaises(errors.DeadlineExceededError):
      rpc_ops.GrpcClient(
          address,
          name="client1",
          list_registered_methods=True,
          timeout_in_ms=1)

    # Create same client again should succeed with
    # list_registered_methods=False. Default timeout for client is 1 ms.
    client = rpc_ops.GrpcClient(
        address, name="client1", list_registered_methods=False, timeout_in_ms=1)

    ensure_server_is_ready(client)
    # Make explicit RPC call, the timeout of 1 ms should lead to
    # deadline exceeded error.

    result_or = client.call(
        "add", [constant_op.constant(20),
                constant_op.constant(30)],
        timeout_in_ms=1)
    self.assertAllEqual(result_or.is_ok(), False)
    error_code, error_message = result_or.get_error()
    self.assertAllEqual(error_code, errors.DEADLINE_EXCEEDED, error_message)

    # Specifying reasonable timeout for call should succeed.
    result_or = client.call(
        "add", [constant_op.constant(20),
                constant_op.constant(30)],
        timeout_in_ms=5000)
    self.assertAllEqual(result_or.is_ok(), True)
    error_code, _ = result_or.get_error()

    # Test timeouts for convenience methods

    # Restart server again with delay to simulate deadline exceeded.
    del server
    server = rpc_ops.GrpcServer(address)
    t = threading.Thread(target=start_server)
    t.start()

    # Client with no default timeout.
    client = rpc_ops.GrpcClient(
        address, name="client2", list_registered_methods=True)

    # Succeeds with reasonable timeout.
    result_or = client.add(
        constant_op.constant(20), constant_op.constant(30), timeout_in_ms=5000)
    self.assertAllEqual(result_or.is_ok(), True)

  def test_async_call_op_wrapper(self):
    # [FUNCTION DOC]
    # Scenario: many asynchronous calls. Registers add-assign and read functions over a captured
    # variable, issues ten add calls of 2 without waiting, then waits on each future's status,
    # and finally reads the variable. Asserts the read call is ok and returns a list holding 20.
    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign_add(a):
      # [FUNCTION DOC]
      # Remote function adding the input to the captured variable; returns nothing.
      v.assign_add(a)

    @eager_def_function.function(input_signature=[])
    def read_var():
      # [FUNCTION DOC]
      # Remote function returning the current value of the captured variable.
      return v.value()

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    server.register("assign_add", assign_add)
    server.register("read_var", read_var)
    server.start()

    client = rpc_ops.GrpcClient(address)

    futures = []
    for _ in range(10):
      futures.append(
          client.call("assign_add",
                      [variables.Variable(2, dtype=dtypes.int64)]))

    for f in futures:
      f.is_ok()

    result_or = client.call(
        "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])

    self.assertAllEqual(result_or.is_ok(), True)
    self.assertAllEqual(result_or.get_value(), [20])

  def test_rpc_call_op_in_tf_function(self):
    # [FUNCTION DOC]
    # Scenario: issue an RPC call from inside a traced function. Registers a multiply function,
    # starts the server, then runs a traced function that calls it with variables 2 and 3 and
    # checks inside that the call is ok and the result is a one-element list. Asserts the
    # traced function returns a list holding 6.

    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def _remote_fn(a, b):
      # [FUNCTION DOC]
      # HELPER: remote function registered by the enclosing test.
      # Returns the product of two scalar integers.
      return math_ops.multiply(a, b)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.GrpcServer(address)

    server_resource.register("remote_fn", _remote_fn)

    server_resource.start()
    client = rpc_ops.GrpcClient(address=address, name="test_client")

    a = variables.Variable(2, dtype=dtypes.int32)
    b = variables.Variable(3, dtype=dtypes.int32)

    @eager_def_function.function
    def call_fn():
      # [FUNCTION DOC]
      # Traced function used by the enclosing test: calls the remote function through the client
      # with the two captured variables, asserts the call is ok and that exactly one value is
      # returned, and returns that list of values.
      result_or = client.call(
          args=[a, b],
          method_name="remote_fn",
          output_specs=[tensor_spec.TensorSpec([], dtypes.int32)])

      self.assertAllEqual(True, result_or.is_ok())
      result = result_or.get_value()
      self.assertEqual(len(result), 1)  # Call returns a list(tensors)
      # TODO(ishark): Shape for output tensor is unknown currently.
      # Add attribute for capturing TensorSpec for output and enable
      # check below:
      # self.assertIsNotNone(result[0].shape.rank)
      return result

    self.assertAllEqual(call_fn(), [6])

  def test_resource_deletion(self):
    # [FUNCTION DOC]
    # Scenario: resources are cleaned up automatically. Creates a server and client, issues calls
    # eagerly and inside traced functions, with and without reading the value, and asserts that
    # destroying each future's status resource afterwards raises a not-found error because it was
    # already deleted. Then deletes the client and the server objects and asserts their handles
    # are likewise gone (not-found on destroy).
    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    server_handle = server._server_handle

    # Test Future resource deletion
    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(input_signature=[])
    def read_var():
      # [FUNCTION DOC]
      # Remote function returning the current value of the captured variable.
      return v.value()

    server.register("read_var", read_var)

    server.start()
    client = rpc_ops.GrpcClient(address)

    client_handle = client._client_handle

    # Check future resource deletion without calling get_value.
    def _create_and_delete_rpc_future():
      # [FUNCTION DOC]
      # HELPER: makes a read call eagerly without reading its value and returns the future's
      # status resource handle, which should be deleted once the future goes out of scope.
      handle = client.call(
          "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])
      return handle._status_or

    @eager_def_function.function
    def _create_and_delete_rpc_future_fn():
      # [FUNCTION DOC]
      # HELPER: traced variant of the eager helper above; makes a read call without reading its
      # value and returns the future's status resource handle.
      handle = client.call(
          "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])
      return handle._status_or

    for _ in range(2):
      handle = _create_and_delete_rpc_future()
      with self.assertRaises(errors.NotFoundError):
        resource_variable_ops.destroy_resource_op(
            handle, ignore_lookup_error=False)

    for _ in range(2):
      handle = _create_and_delete_rpc_future_fn()
      with self.assertRaises(errors.NotFoundError):
        resource_variable_ops.destroy_resource_op(
            handle, ignore_lookup_error=False)

    # Check future resource deletion with calling get_value.
    def _create_and_delete_with_future():
      # [FUNCTION DOC]
      # HELPER: makes a read call eagerly, reads its value, and returns the future's status
      # resource handle, which should be deleted after the value was consumed.
      handle = client.call(
          "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])
      status_or_handle = handle._status_or
      handle.get_value()
      return status_or_handle

    # Check future resource deletion with calling get_value with tf.function.
    @eager_def_function.function
    def _create_and_delete_with_future_fn():
      # [FUNCTION DOC]
      # HELPER: traced variant of the previous helper; makes a read call, reads its value and
      # returns the future's status resource handle.
      handle = client.call(
          "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])
      status_or_handle = handle._status_or
      handle.get_value()
      return status_or_handle

    for _ in range(2):
      resource_handle = _create_and_delete_with_future()
      with self.assertRaises(errors.NotFoundError):
        resource_variable_ops.destroy_resource_op(
            resource_handle, ignore_lookup_error=False)

    for _ in range(2):
      resource_handle = _create_and_delete_with_future_fn()
      with self.assertRaises(errors.NotFoundError):
        resource_variable_ops.destroy_resource_op(
            resource_handle, ignore_lookup_error=False)

    # Test server client resource gets deleted.
    del client
    with self.assertRaises(errors.NotFoundError):
      resource_variable_ops.destroy_resource_op(
          client_handle, ignore_lookup_error=False)

    # Test server server resource gets deleted.
    del server
    with self.assertRaises(errors.NotFoundError):
      resource_variable_ops.destroy_resource_op(
          server_handle, ignore_lookup_error=False)

  def test_rpc_error(self):
    # [FUNCTION DOC]
    # Scenario: error reporting. Registers add-assign and read functions and checks normal calls
    # work (generic and convenience forms, values 2 then 4). Calling add-assign with no
    # arguments must return a not-ok result with the invalid-argument code. After deleting the
    # server, a blocking call with a 1 ms timeout must raise a deadline-exceeded error.
    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign_add(a):
      # [FUNCTION DOC]
      # Remote function adding the input to the captured variable; returns nothing.
      v.assign_add(a)

    @eager_def_function.function(input_signature=[])
    def read_var():
      # [FUNCTION DOC]
      # Remote function returning the current value of the captured variable.
      return v.value()

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    server.register("assign_add", assign_add)
    server.register("read_var", read_var)
    server.start()

    client = rpc_ops.GrpcClient(address, list_registered_methods=True)

    # confirm it works as expected when arguments are passed.
    result_or = client.call("assign_add",
                            [variables.Variable(2, dtype=dtypes.int64)])
    self.assertAllEqual(result_or.is_ok(), True)
    result_or = client.call(
        "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])
    self.assertAllEqual(result_or.is_ok(), True)
    self.assertAllEqual(result_or.get_value(), [2])
    result_or = client.assign_add(variables.Variable(2, dtype=dtypes.int64))
    self.assertAllEqual(True, result_or.is_ok())

    result_or = client.read_var()
    self.assertAllEqual(True, result_or.is_ok())
    self.assertAllEqual(result_or.get_value(), 4)

    # Fails with invalid argument error when no arguments are passed.
    result_or = client.call("assign_add")
    self.assertAllEqual(result_or.is_ok(), False)
    error_code, _ = result_or.get_error()
    self.assertAllEqual(error_code, errors.INVALID_ARGUMENT)

    del server
    with self.assertRaises(errors.DeadlineExceededError):
      _ = client.assign_add_blocking(
          variables.Variable(2, dtype=dtypes.int64), timeout_in_ms=1)

  def test_captured_inputs(self):
    # [FUNCTION DOC]
    # Scenario: registered functions capture a variable. Calls add-assign with 2 twice and then
    # reads the variable. Asserts all calls are ok and the read returns a list holding 4.
    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign_add(a):
      # [FUNCTION DOC]
      # Remote function adding the input to the captured variable; returns nothing.
      v.assign_add(a)

    @eager_def_function.function(input_signature=[])
    def read_var():
      # [FUNCTION DOC]
      # Remote function returning the current value of the captured variable.
      return v.value()

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    server.register("assign_add", assign_add)
    server.register("read_var", read_var)

    server.start()

    client = rpc_ops.GrpcClient(address)

    result_or = client.call("assign_add",
                            [variables.Variable(2, dtype=dtypes.int64)])
    self.assertAllEqual(result_or.is_ok(), True)
    result_or = client.call("assign_add",
                            [variables.Variable(2, dtype=dtypes.int64)])
    self.assertAllEqual(result_or.is_ok(), True)
    result_or = client.call(
        "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])

    self.assertAllEqual(result_or.is_ok(), True)
    self.assertAllEqual(result_or.get_value(), [4])

  def test_register_method_twice(self):
    # [FUNCTION DOC]
    # Scenario: duplicate method names. Registers one function under the name assign, then tries
    # to register a different function under the same name. Asserts an invalid-argument error
    # saying the name is already registered. The server is never started.
    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign_add(a):
      # [FUNCTION DOC]
      # Remote function adding the input to the captured variable; returns nothing.
      v.assign_add(a)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign(a):
      # [FUNCTION DOC]
      # Remote function overwriting the captured variable with the input; returns nothing.
      v.assign(a)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    server.register("assign", assign_add)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "assign is already registered."):
      # Reusing the same error name.
      server.register("assign", assign)

  def test_tf_function_register_without_input_signature(self):
    # [FUNCTION DOC]
    # Scenario: registering functions lacking an input signature. Registering a function with an
    # argument but no signature must raise a value error saying the input signature is not
    # specified. Registering a function without arguments and without a signature must succeed.
    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function
    def assign(a):
      # [FUNCTION DOC]
      # Remote function overwriting the captured variable with the input; returns nothing. It has
      # no input signature, which makes registration fail.
      v.assign(a)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    with self.assertRaisesRegex(
        ValueError, "Input signature not specified for the function."):
      server.register("assign", assign)

    # Register without input signature should work for functions without input
    # args.
    @eager_def_function.function
    def read_var():
      # [FUNCTION DOC]
      # Remote function without inputs returning the current value of the captured variable.
      return v.value()

    server.register("read_var", read_var)

  def test_multi_device_resource(self):
    # [FUNCTION DOC]
    # Scenario: a function captures a queue resource living on the second logical CPU while the
    # server and client are created under the first CPU. Calls the populate function through the
    # client, then asserts that dequeuing yields every random element in order.
    elements = np.random.randint(100, size=[200])

    with ops.device("/device:CPU:1"):
      queue = data_flow_ops.FIFOQueue(200, dtypes.int64, shapes=[])

    @eager_def_function.function()
    def populate_queue():
      # [FUNCTION DOC]
      # Remote function that enqueues all elements into the captured queue and closes it.
      queue.enqueue_many(elements)
      queue.close()

    with ops.device("/device:CPU:0"):
      port = portpicker.pick_unused_port()
      address = "localhost:{}".format(port)
      server = rpc_ops.GrpcServer(address)
      server.register("populate_queue", populate_queue)
      server.start()

      client = rpc_ops.GrpcClient(address, list_registered_methods=True)
      client.populate_queue()

    for e in elements:
      self.assertAllEqual(e, queue.dequeue())

  def test_queue_resource(self):
    # [FUNCTION DOC]
    # Scenario: a registered function captures a queue resource on the default device. Calls the
    # populate function remotely, then asserts that dequeuing yields every random element in order.
    elements = np.random.randint(100, size=[200])
    queue = data_flow_ops.FIFOQueue(200, dtypes.int64, shapes=[])

    @eager_def_function.function()
    def populate_queue():
      # [FUNCTION DOC]
      # Remote function that enqueues all elements into the captured queue and closes it.
      queue.enqueue_many(elements)
      queue.close()

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    server.register("populate_queue", populate_queue)
    server.start()

    client = rpc_ops.GrpcClient(address, list_registered_methods=True)
    client.populate_queue()

    for e in elements:
      self.assertAllEqual(e, queue.dequeue())

  def test_multi_device_resource_cpu(self):
    # [FUNCTION DOC]
    # Scenario: a variable lives on the second logical CPU while the server and client run under
    # the first. Calls add-assign with 2 through the client. Asserts the call is ok and the
    # variable ends at 2.
    with ops.device("/device:cpu:1"):
      v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign_add(a):
      # [FUNCTION DOC]
      # Remote function adding the input to the captured variable; returns nothing.
      v.assign_add(a)

    with ops.device("/device:CPU:0"):
      port = portpicker.pick_unused_port()
      address = "localhost:{}".format(port)
      server = rpc_ops.GrpcServer(address)
      server.register("assign_add", assign_add)
      server.start()

      client = rpc_ops.GrpcClient(address, list_registered_methods=True)
      result_or = client.assign_add(variables.Variable(2, dtype=dtypes.int64))
      self.assertAllEqual(result_or.is_ok(), True)

    self.assertAllEqual(v, 2)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
