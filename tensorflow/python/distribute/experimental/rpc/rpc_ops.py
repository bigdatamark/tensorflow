# [MODULE DOC]
# Public RPC API for TensorFlow distribution: a server on which compiled functions are
# registered and exposed over the network, a client that invokes them remotely, and a result
# object representing the pending or finished outcome of a call. Exposes the abstract Server and
# Client base classes (exported publicly with static create factories, only the gRPC layer is
# supported), the concrete gRPC server and client, the status-or-result class, and two helpers
# that serialize input and output type specifications of a concrete function. Built on the
# generated RPC kernel ops and the RPC protocol buffer messages. No import-time side effects
# other than the public API export registrations.
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
"""Module to expose RPC APIs in tensorflow."""

from typing import Optional, Sequence, Union

import tensorflow.distribute.experimental.rpc.kernels.gen_rpc_ops as gen_rpc_ops
from tensorflow.distribute.experimental.rpc.proto import tf_rpc_service_pb2 as rpc_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import none_tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


def get_output_specs_from_function(func: tf_function.ConcreteFunction):
  # [FUNCTION DOC]
  # HELPER: serializes a function's output signature for server registration.
  # Takes a concrete function. Converts each element of its structured outputs to a type
  # specification, encodes the whole structure as a protocol buffer message and returns it as a
  # serialized byte string.
  output_specs = nest.map_structure(type_spec.type_spec_from_value,
                                    func.structured_outputs)
  output_specs_proto = nested_structure_coder.encode_structure(output_specs)
  return output_specs_proto.SerializeToString()


def get_input_specs_from_function(func: tf_function.ConcreteFunction):
  # [FUNCTION DOC]
  # HELPER: serializes a function's input signature for server registration.
  # Takes a concrete function. Takes the positional argument specifications of its structured
  # input signature (ignoring keyword ones), encodes them as a protocol buffer message and returns
  # it as a serialized byte string.
  arg_specs, _ = func.structured_input_signature
  arg_specs_proto = nested_structure_coder.encode_structure(arg_specs)
  return arg_specs_proto.SerializeToString()


@tf_export("distribute.experimental.rpc.Server", v1=[])
class Server(object):
  # [CLASS DOC]
  # Abstract base class (publicly exported) for an RPC server that exposes registered compiled
  # functions. Has no state itself. Instances are obtained through the static create factory,
  # which returns the gRPC subclass. The register and start methods here only raise.
  """A Server base class for accepting RPCs for registered tf.functions.

    Functions can be registered on the server and are exposed via RPCs.
  """

  @staticmethod
  def create(rpc_layer, address):
    # [FUNCTION DOC]
    # Static factory. Parameters: rpc_layer, a string naming the communication layer (only grpc
    # is accepted); address, a string host and port where the server will listen. Raises a value
    # error for any other layer. Returns a new gRPC server (which allocates a server resource,
    # see its constructor for constraints).
    """Create TF RPC server at given address.

    Args:
      rpc_layer: Communication layer between client and server. Only "grpc" rpc
        layer is supported at the moment.
      address: Address where RPC server is hosted.

    Returns:
      An instance of `tf.distribute.experimental.rpc.Server` class.

    Raises:
        A ValueError if rpc_layer other than "grpc" is used. Only GRPC
        is supported at the moment.

    Example usage:

      >>> import portpicker
      >>> @tf.function(input_signature=[
      ...      tf.TensorSpec([], tf.int32),
      ...      tf.TensorSpec([], tf.int32)])
      ... def remote_fn(a, b):
      ...   return tf.add(a, b)

      >>> port = portpicker.pick_unused_port()
      >>> address = "localhost:{}".format(port)
      >>> server = tf.distribute.experimental.rpc.Server.create("grpc", address)
      >>> server.register("addition", remote_fn)
      >>> server.start()

    """
    if rpc_layer != "grpc":
      raise ValueError("Only GRPC backend is supported at the moment.")
    return GrpcServer(address=address)

  def register(self, method_name: str,
               func: Union[def_function.Function,
                           tf_function.ConcreteFunction]):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error message refers to a create_server method that does not exist (the
    # factory is named create), and the two concatenated message pieces lack a space, so the text
    # reads "create aconcrete subclass".
    # Abstract method: parameters are the method name string and the function to register. Always
    # raises a not-implemented error; the gRPC subclass provides the real behavior.
    """Method for registering tf.function on server.

    Registered methods can be invoked remotely from clients.

    Args:
      method_name: Name of the tf.function. Clients use this method_name to make
        RPCs.
      func: A `tf.function` or ConcreteFunction to register.
    """
    raise NotImplementedError("Please use create_server method to create a"
                              "concrete subclass of Server.")

  def start(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error message refers to a create_server method that does not exist (the
    # factory is named create), and the two concatenated message pieces lack a space, so the text
    # reads "create aconcrete subclass".
    # Abstract method with no parameters. Always raises a not-implemented error; the gRPC
    # subclass provides the real behavior.
    """Starts the RPC server on provided address.

     Server listens for new requests from client, once it is started.
    """
    raise NotImplementedError("Please use create_server method to create a"
                              "concrete subclass of Server.")


@tf_export("distribute.experimental.rpc.Client", v1=[])
class Client(object):
  # [CLASS DOC]
  # Abstract base class (publicly exported) for an RPC client that invokes functions registered on
  # a remote server. Has no state itself. Instances are obtained through the static create factory,
  # which returns the gRPC subclass; the call method is abstract.
  """Client class for invoking RPCs to the server."""

  @staticmethod
  def create(rpc_layer, address, name="", timeout_in_ms=0):
    # [FUNCTION DOC]
    # Static factory. Parameters: rpc_layer, string naming the layer (only grpc accepted);
    # address, string address of the server; name, optional client name (default empty), used as
    # the shared resource name; timeout_in_ms, default per-call timeout in milliseconds (zero means
    # none). Raises a value error for another layer. If executing eagerly, the client asks the
    # server for its registered methods and adds convenience methods; otherwise it does not.
    # Returns a new gRPC client.
    """Create TF RPC client to connect to the given address.

    Args:
      rpc_layer: Communication layer between client and server. Only "grpc" rpc
        layer is supported at the moment.
      address: Address of the server to connect the RPC client to.
      name: Name of the RPC Client. You can create multiple clients connecting
        to same server and distinguish them using different names.
      timeout_in_ms: The default timeout to use for outgoing RPCs from client. 0
        indicates no timeout. Exceeding timeout during RPC will raise
        DeadlineExceeded error.

    Returns:
      An instance of `tf.distribute.experimental.rpc.Client` with the following
      dynamically added methods for eagerly created clients:
        * `Registered methods` e.g. multiply(**args):
            If Client is created when executing eagerly, client will request the
            list of registered methods from server during client creation.
            The convenience methods for RPCs will be dynamically added to the
            created Client instance.

            For example, when a server has method "multiply" registered, the
            client object created in eager mode will have 'multiply' method
            available. Users can use client.multiply(..) to make RPC, instead of
            client.call("multiply", ...)

            Both "call" and "multiply" methods are non-blocking i.e. they return
            a StatusOrResult object which should be used to wait for getting
            value or error.

            Along with the above, blocking versions of the registered
            methods are also dynamically added to client instance.
            e.g. multiply_blocking(**args). These methods block till the RPC is
            finished and return response for successful RPC. Otherwise raise
            exception.

            These methods are not available when Client is created inside a
            tf.function.

    Raises:
        A ValueError if rpc_layer other than "grpc" is used. Only GRPC
          is supported at the moment.
        A DeadlineExceeded exception in eager mode if timeout exceeds while
          creating and listing client methods.

    Example usage:
      >>> # Have server already started.
      >>> import portpicker
      >>> @tf.function(input_signature=[
      ...      tf.TensorSpec([], tf.int32),
      ...      tf.TensorSpec([], tf.int32)])
      ... def remote_fn(a, b):
      ...   return tf.add(a, b)

      >>> port = portpicker.pick_unused_port()
      >>> address = "localhost:{}".format(port)
      >>> server = tf.distribute.experimental.rpc.Server.create("grpc", address)
      >>> server.register("addition", remote_fn)
      >>> server.start()

      >>> # Start client
      >>> client = tf.distribute.experimental.rpc.Client.create("grpc",
      ...      address=address, name="test_client")

      >>> a = tf.constant(2, dtype=tf.int32)
      >>> b = tf.constant(3, dtype=tf.int32)

      >>> result = client.call(
      ...    args=[a, b],
      ...    method_name="addition",
      ...    output_specs=tf.TensorSpec((), tf.int32))

      >>> if result.is_ok():
      ...   result.get_value()

      >>> result = client.addition(a, b)

      >>> if result.is_ok():
      ...   result.get_value()

      >>> value = client.addition_blocking(a, b)
    """
    if rpc_layer != "grpc":
      raise ValueError("Only GRPC backend is supported at the moment.")
    if context.executing_eagerly():
      list_registered_methods = True
    else:
      list_registered_methods = False
    return GrpcClient(
        address=address,
        name=name,
        list_registered_methods=list_registered_methods,
        timeout_in_ms=timeout_in_ms)

  def call(self,
           method_name: str,
           args: Optional[Sequence[core_tf_types.Tensor]] = None,
           output_specs=None,
           timeout_in_ms=0):
    # [FUNCTION DOC]
    # Abstract method. Parameters: method_name, the registered remote method; args, optional list
    # of tensor arguments; output_specs, optional type specification of the outputs; timeout_in_ms,
    # per-call timeout (zero uses the client default). Always raises a not-implemented error here;
    # the gRPC subclass provides the behavior.
    """Method for making RPC calls to remote server.

    This invokes RPC to the server, executing the registered method_name
    remotely.
    Args:
      method_name: Remote registered method to invoke
      args: List of arguments for the registered method.
      output_specs: Output specs for the output from method.
         For example, if tf.function is: @tf.function(input_signature=[
           tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32) ])
          def multiply_fn(a, b): return tf.math.multiply(a, b)
        output_spec is: tf.TensorSpec((), tf.int32)  If you have access to TF
          Function, the output specs can be generated
       from tf.function by calling: output_specs =
         tf.nest.map_structure(tf.type_spec_from_value,
         tf_function.get_concrete_function().structured_outputs  If output_specs
         are not provided, flattened list of tensors will be returned in
         response.
      timeout_in_ms: Timeout for this call. If 0, default client timeout will be
        used.

    Returns:
      An instance of `StatusOrResult` class with the following available
      methods.
        * `is_ok()`:
            Returns True of RPC was successful.
        * `get_error()`:
            Returns TF error_code and error message for the RPC.
        * `get_value()`:
            Returns the returned value from remote TF function execution
            when RPC is successful.

      Calling any of the above methods will block till RPC is completed and
      result is available.
    """
    raise NotImplementedError("Must be implemented in inherited classes.")


class GrpcServer(Server):
  # [CLASS DOC]
  # Concrete RPC server using gRPC. Holds a server resource handle created by the generated op
  # and a deleter object that releases the resource when the server object is destroyed. Created
  # via the base class factory; register adds functions and start begins serving.
  """GrpcServer object encapsulates a resource with GRPC server.

    Functions can be registered locally and are exposed via RPCs.
    Example:
    ```
    server = rpc_ops.GrpcServer("host:port")
    @tf.function
    def add(a, b):
      return a + b

    server.register("add", add)
    server.start()
    ```
  """

  def __init__(self, address: str):
    # [FUNCTION DOC]
    # Creates the server resource for the given address string (host and port). Must run
    # eagerly: in that case an automatic deleter is attached to release the resource later.
    # Otherwise raises a not-implemented error saying to create the server outside a compiled
    # function (note the resource op has already been created by then).
    self._server_handle = gen_rpc_ops.rpc_server(address)
    if context.executing_eagerly():
      self._handle_deleter = resource_variable_ops.EagerResourceDeleter(
          handle=self._server_handle, handle_device=self._server_handle.device)
    else:
      raise NotImplementedError("Please create the server outside tf.function.")

  def register(self, method_name: str,
               func: Union[def_function.Function,
                           tf_function.ConcreteFunction]):
    # [FUNCTION DOC]
    # Registers a function on the server under a method name. Parameters: method_name, a string
    # clients use to call it; func, either a traced function wrapper or a concrete function. For a
    # traced wrapper that has named arguments an input signature must be defined, else a value
    # error is raised; its concrete function is then obtained and registered. For a concrete
    # function it is registered directly. Registration passes the captured inputs, the serialized
    # input and output specifications and the function to the server op. Any other type (such as
    # a plain Python function) raises a value error. Returns nothing.
    """Method for registering functions."""

    if isinstance(func, def_function.Function):
      if func.function_spec.arg_names:
        if func.input_signature is None:
          raise ValueError("Input signature not specified for the function.")
      concrete_fn = func.get_concrete_function()
      gen_rpc_ops.rpc_server_register(
          self._server_handle,
          method_name=method_name,
          captured_inputs=concrete_fn.captured_inputs,
          input_specs=get_input_specs_from_function(concrete_fn),
          output_specs=get_output_specs_from_function(concrete_fn),
          f=concrete_fn)
    elif isinstance(func, tf_function.ConcreteFunction):
      gen_rpc_ops.rpc_server_register(
          self._server_handle,
          method_name=method_name,
          captured_inputs=func.captured_inputs,
          input_specs=get_input_specs_from_function(func),
          output_specs=get_output_specs_from_function(func),
          f=func)
    else:
      # Python functions
      # TODO(b/186762191): Add an implementation to support python functions.
      raise ValueError("Only TF functions are supported with Register method")

  def start(self):
    # [FUNCTION DOC]
    # Starts the gRPC server so it begins accepting requests, using the server op on the stored
    # handle. Returns nothing.
    """Starts GRPC server."""
    gen_rpc_ops.rpc_server_start(self._server_handle)


class GrpcClient(Client):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says convenience methods are named with an async suffix, but the
  # code adds a method with the registered name plus a second one with a blocking suffix.
  # Concrete RPC client using gRPC. Holds a client resource handle with its deleter, the server
  # address, and a registry mapping each discovered remote method name to its output specs. When
  # created eagerly with method listing enabled, it adds dynamic per-method attributes (non-blocking
  # and blocking variants) to the instance.
  """Client wrapper to connect to remote RPC server using GRPC.

  If Client is created with (list_registered_methods=True):
  1. Input and output specs for the methods till this point will be fetched from
  Server.
  2. convenience methods are added to invoke registered methods directly from
  client.
  For example:
    For call a server method `add`
    client.add(a, b) or client.add_async(a, b) can be used instead of
    client.call(args=[a,b], output_specs=[..])

  Prerequisite for using list_registered_methods=True:
   1. Server should be already started with the registered methods.
   2. Client must be created in Eager mode.
  """

  def __init__(self,
               address: str,
               name: str = "",
               list_registered_methods=False,
               timeout_in_ms=0):
    # [FUNCTION DOC]
    # Creates the client. Parameters: address, server address string; name, optional shared
    # resource name (default empty); list_registered_methods, boolean (default false) asking the
    # server to list its registered methods; timeout_in_ms, default call timeout (zero means
    # none). Creates the client resource via the generated op, then requires eager execution
    # (else raises a not-implemented error) and attaches a deleter. Stores the address and
    # builds the method registry: for each registered method returned by the server it decodes
    # the serialized message, decodes the input and output specs, records the output specs and
    # adds convenience methods to this instance via the private add-method helper with a generated
    # documentation string. Can raise a deadline-exceeded error if listing times out.
    self._client_handle, methods = gen_rpc_ops.rpc_client(
        shared_name=name,
        server_address=address,
        list_registered_methods=list_registered_methods,
        timeout_in_ms=timeout_in_ms)
    if context.executing_eagerly():
      self._handle_deleter = resource_variable_ops.EagerResourceDeleter(
          handle=self._client_handle, handle_device=self._client_handle.device)
    else:
      raise NotImplementedError(
          "Client creation is supported only in eager mode.")
    self._server_address = address
    self._method_registry = {}
    for method in methods.numpy():
      m = rpc_pb2.RegisteredMethod()
      m.ParseFromString(method)
      output_specs = nested_structure_coder.decode_proto(m.output_specs)
      input_specs = nested_structure_coder.decode_proto(m.input_specs)
      self._method_registry[m.method] = output_specs
      # TODO(ishark): Perhaps doc string can also be taken as input during
      # function registration.
      doc_string = "RPC Call for " + m.method + " method to server " + address
      self._add_method(m.method, output_specs, input_specs, self._client_handle,
                       doc_string)

  def _add_method(self, method_name, output_specs, input_specs, client_handle,
                  doc_string):
    # [FUNCTION DOC]
    # HELPER: dynamically attaches per-method convenience callables to the client instance.
    # Parameters: method_name, remote method name; output_specs and input_specs, decoded type
    # specifications; client_handle, the client resource; doc_string, text used as the new
    # callables' documentation. Sets an attribute named as the method (non-blocking: returns a
    # status-or-result object) and another with a blocking suffix (waits, returns the value or
    # raises the mapped error). Overwrites any existing attribute of the same name, even the call
    # method. Returns nothing.
    """Method to add RPC methods to the client object."""

    def validate_and_get_flat_inputs(*args):
      # [FUNCTION DOC]
      # KNOWN ISSUE: the null check is dead code, because variadic arguments are always a
      # (possibly empty) tuple and never null.
      # Takes positional arguments; if input specs are known, asserts the arguments have the same
      # nested structure as them (raises on mismatch). Returns the flattened list of arguments.
      if args is None:
        args = []
      if input_specs:
        nest.assert_same_structure(args, input_specs)
      flat_inputs = nest.flatten(args)
      return flat_inputs

    def call_wrapper(*args, timeout_in_ms=0):
      # [FUNCTION DOC]
      # Non-blocking convenience call. Takes the method's positional arguments and an optional
      # keyword timeout in milliseconds (zero uses the default). Validates and flattens the
      # arguments, issues the RPC and returns a status-or-result object using the known output
      # specs, without waiting.
      status_or, deleter = gen_rpc_ops.rpc_call(
          client_handle,
          args=validate_and_get_flat_inputs(*args),
          method_name=method_name,
          timeout_in_ms=timeout_in_ms)
      return StatusOrResult(status_or, deleter, output_specs)

    def call_blocking_wrapper(*args, timeout_in_ms=0):
      # [FUNCTION DOC]
      # Blocking convenience call. Same inputs as the non-blocking variant. Issues the RPC and
      # waits; if it succeeded returns the value, otherwise raises the TensorFlow exception type
      # matching the returned error code with the returned error message.
      status_or, deleter = gen_rpc_ops.rpc_call(
          client_handle,
          args=validate_and_get_flat_inputs(*args),
          method_name=method_name,
          timeout_in_ms=timeout_in_ms)
      status_or = StatusOrResult(status_or, deleter, output_specs)
      if status_or.is_ok():
        return status_or.get_value()
      else:
        error_code, error_msg = status_or.get_error()
        raise errors.exception_type_from_error_code(error_code.numpy())(
            None, None, error_msg.numpy())

    setattr(self, method_name, call_wrapper)
    call_wrapper.__doc__ = doc_string

    blocking_method_name = method_name + "_blocking"
    setattr(self, blocking_method_name, call_blocking_wrapper)
    call_blocking_wrapper.__doc__ = doc_string

  def call(self,
           method_name: str,
           args: Optional[Sequence[core_tf_types.Tensor]] = None,
           output_specs=None,
           timeout_in_ms=0):
    # [FUNCTION DOC]
    # Issues an RPC without waiting. Parameters: method_name, the registered method; args,
    # optional list of arguments (treated as empty if null, flattened without structure
    # checks); output_specs, optional output type specification; timeout_in_ms, per-call timeout
    # (zero uses the client default). Returns a status-or-result object.
    """Method to invoke remote registered functions on the connected server.

    Server should be started before making an RPC Call.

    Args:
      method_name: Registered method to invoke on Server.
      args: Input arguments for the method.
      output_specs: Output specs for the output from method.
      timeout_in_ms: Timeout for this call. If 0, default client timeout will be
       used.

    Returns:
      StatusOrResult object. This function issues the RPC call to server, it
      does not block for the duration of RPC. Please call is_ok, get_error or
      get_value methods on the returned object to blocked till RPC finishes.
    """
    if args is None:
      args = []
    status_or, deleter = gen_rpc_ops.rpc_call(
        self._client_handle,
        args=nest.flatten(args),
        method_name=method_name,
        timeout_in_ms=timeout_in_ms)
    return StatusOrResult(status_or, deleter, output_specs)


class StatusOrResult(object):
  # [CLASS DOC]
  # Handle for the outcome of one asynchronous RPC. Holds the future resource, its deleter, the
  # output specs used to decode the value, and a cached error code and message (filled on first
  # status check). Created by the client call methods. Its accessors block until the RPC finishes.
  # The destructor releases the future resource.
  """Class representing result and status from RPC Call."""

  def __init__(self, status_or, deleter, output_specs=None):
    # [FUNCTION DOC]
    # Stores the future resource handle, the deleter resource, and the optional output specs
    # (default null) and initializes the cached error code and message to null.
    self._status_or = status_or
    self._output_specs = output_specs
    self._deleter = deleter
    self._error_code: dtypes.int64 = None
    self._error_message: dtypes.string = None

  def _check_status(self):
    # [FUNCTION DOC]
    # HELPER: lazily fetches and caches the RPC status for the accessor methods.
    # If the error code has not been fetched yet, blocks until the RPC completes and stores the
    # error code and message tensors; otherwise does nothing. Returns nothing.
    if self._error_code is None:
      self._error_code, self._error_message = gen_rpc_ops.rpc_check_status(
          self._status_or)

  def __del__(self):
    # [FUNCTION DOC]
    # Destructor. Deletes the future resource through the generated op, in the same execution mode
    # (eager or graph) as the current one, using eager mode if currently executing eagerly and
    # graph mode otherwise. Side effect: frees the resource. Note it would fail if construction
    # did not complete since attributes would be missing.
    # Make sure the resource is deleted in the same mode as it was created in.
    if context.executing_eagerly():
      with context.eager_mode():
        gen_rpc_ops.delete_rpc_future_resource(
            handle=self._status_or, deleter=self._deleter)
    else:
      with context.graph_mode():
        gen_rpc_ops.delete_rpc_future_resource(
            handle=self._status_or, deleter=self._deleter)

  def is_ok(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns a boolean, but it returns a scalar boolean
    # tensor produced by comparing the error code with zero.
    # Blocks until the RPC completes, then returns whether the error code equals zero.
    """Returns True if RPC is successful, otherwise returns False.

    This call will block for RPC result.
    """
    self._check_status()
    return math_ops.equal(self._error_code,
                          constant_op.constant(0, dtype=dtypes.int64))

  def get_error(self):
    # [FUNCTION DOC]
    # Blocks until the RPC completes and returns a pair: the error code tensor (64-bit integer)
    # and the error message tensor (string). Code zero means success.
    """Returns (TF Error Code, Error Message) from RPC Response.

    This call will block for RPC result.
    """
    self._check_status()
    return self._error_code, self._error_message

  def get_value(self):
    # [FUNCTION DOC]
    # Blocks until the RPC completes, then fetches the response. If no output specs were given
    # or they are the none-tensor spec, requests no outputs and returns nothing (null). Otherwise
    # requests tensors with the data types of the flattened output specs and packs them into the
    # nested structure of the output specs, which it returns. Does not itself check that the RPC
    # succeeded; the underlying op reports errors.
    """Returns the returned response value from RPC Call when RPC is successful.

      The returned value is tensors in the output_specs format as returned from
      the RPC call


    This call will block for RPC result.
    """

    self._check_status()
    if self._output_specs is None or isinstance(self._output_specs,
                                                none_tensor.NoneTensorSpec):
      flat_output_dtypes = []
      return_none = True
    else:
      return_none = False
      flat_output_dtypes = [s.dtype for s in nest.flatten(self._output_specs)]

    result = gen_rpc_ops.rpc_get_value(self._status_or, Tout=flat_output_dtypes)
    if return_none:
      return None
    else:
      return nest.pack_sequence_as(self._output_specs, result)
