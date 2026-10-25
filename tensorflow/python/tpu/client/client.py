# [MODULE DOC]
# Cloud TPU client: resolves a TPU name to network endpoints and queries TPU state, health,
# symptoms, runtime version and accelerator type, either through the Cloud TPU API or, for direct
# grpc addresses, without the API. Also can wait for health and switch the TPU software version.
# Exposes the Client class plus private helpers for environment variables and GCE metadata.
# Import-time side effects: tries to import the optional Google API client and oauth2client
# libraries (recording whether they are installed) and defines two boolean command-line flags,
# runtime_oom_exit and hbm_oom_exit, which control whether recent OOM symptoms mark the TPU as
# not recoverable.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cloud TPU Client."""

from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib

from absl import flags

_GOOGLE_API_CLIENT_INSTALLED = True
try:
  from googleapiclient import discovery  # pylint: disable=g-import-not-at-top
  from oauth2client import client  # pylint: disable=g-import-not-at-top
except ImportError:
  _GOOGLE_API_CLIENT_INSTALLED = False

FLAGS = flags.FLAGS

flags.DEFINE_bool('runtime_oom_exit', True,
                  'Exit the script when the TPU runtime is OOM.')
flags.DEFINE_bool('hbm_oom_exit', True,
                  'Exit the script when the TPU HBM is OOM.')

_GKE_ENV_VARIABLE = 'KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS'
_DEFAULT_TPUCONFIG_VARIABLE = 'TPU_CONFIG'
_ENDPOINTS_SEPARATOR = ','
_DEFAULT_ENV_VARIABLE = 'TPU_NAME'
_DISCOVERY_SERVICE_URL_ENV_VARIABLE = 'TPU_API_DISCOVERY_URL'
_GCE_METADATA_URL_ENV_VARIABLE = 'GCE_METADATA_IP'
_GCE_METADATA_ENDPOINT_ENV_VARIABLE = 'GCE_METADATA_HOST'
_DEFAULT_ENDPOINT_PORT = '8470'
_OOM_EVENT_COOL_TIME_SEC = 90
_VERSION_SWITCHER_ENDPOINT = 'http://{}:8475/requestversion'


def _utcnow():
  # [FUNCTION DOC]
  # HELPER: time source for OOM age checks, separated so tests can replace it.
  # Takes nothing and returns the current UTC time as a naive datetime. No side effects.
  """A wrapper function around datetime.datetime.utcnow.

  This function is created for unit testing purpose. It's not easy to do
  StubOutWithMock with datetime.datetime package.

  Returns:
    datetime.datetime
  """
  return datetime.datetime.utcnow()


def _environment_discovery_url():
  # [FUNCTION DOC]
  # HELPER: reads the optional API discovery URL override for the client.
  # Returns the value of the discovery URL environment variable, or nothing if it is not set.
  return os.environ.get(_DISCOVERY_SERVICE_URL_ENV_VARIABLE)


def _gce_metadata_endpoint():
  # [FUNCTION DOC]
  # HELPER: builds the base URL of the GCE metadata server for metadata requests.
  # Uses the metadata host environment variable if set and non-empty, else the metadata IP
  # environment variable, else the default metadata host name. Returns the host prefixed with
  # the http scheme.
  endpoint = os.environ.get(_GCE_METADATA_ENDPOINT_ENV_VARIABLE)
  if not endpoint:
    endpoint = os.environ.get(
        _GCE_METADATA_URL_ENV_VARIABLE, 'metadata.google.internal'
    )
  return 'http://' + endpoint


def _request_compute_metadata(path):
  # [FUNCTION DOC]
  # HELPER: fetches one value from the GCE metadata server (project id, zone, local ip).
  # Input: path, the metadata path relative to the version one compute metadata root. Sends an
  # HTTP request with the Google metadata flavor header and returns the response body as text.
  # Side effect: network access; network errors from the HTTP library propagate to the caller.
  req = urllib.request.Request(
      '%s/computeMetadata/v1/%s' % (_gce_metadata_endpoint(), path),
      headers={'Metadata-Flavor': 'Google'})
  resp = urllib.request.urlopen(req)
  return _as_text(resp.read())


def _environment_var_to_network_endpoints(endpoints):
  # [FUNCTION DOC]
  # HELPER: parses a comma separated list of TPU addresses into endpoint records.
  # Input: endpoints, a string of addresses, each optionally prefixed with the grpc scheme and
  # optionally followed by a colon and port. Yields, per address, a mapping with the ip address
  # and the port (the default TPU port when none is given). No side effects.
  """Yields a dict with ip address and port."""
  for endpoint in endpoints.split(','):
    grpc_prefix = 'grpc://'
    if endpoint.startswith(grpc_prefix):
      endpoint = endpoint.split(grpc_prefix)[1]
    parts = endpoint.split(':')
    ip_address = parts[0]
    port = _DEFAULT_ENDPOINT_PORT
    if len(parts) > 1:
      port = parts[1]
    yield {
        'ipAddress': ip_address,
        'port': port
    }


def _get_tpu_node_config():
  # [FUNCTION DOC]
  # HELPER: reads the TPU configuration from the TPU config environment variable.
  # Returns the parsed JSON mapping if the variable is set and non-empty, otherwise nothing.
  # Raises a JSON decoding error if the variable holds invalid JSON.
  tpu_config_env = os.environ.get(_DEFAULT_TPUCONFIG_VARIABLE)
  if tpu_config_env:
    return json.loads(tpu_config_env)
  return None


def _get_tpu_name(tpu):
  # [FUNCTION DOC]
  # HELPER: determines the TPU name for the client.
  # Input: tpu, an explicit name (returned as is if non-empty). Otherwise checks the GKE
  # endpoints environment variable and then the TPU name environment variable, in that order,
  # returning the first one present. Returns nothing if none is available.
  if tpu:
    return tpu

  for e in [_GKE_ENV_VARIABLE, _DEFAULT_ENV_VARIABLE]:
    if e in os.environ:
      return os.environ[e]
  return None


def _as_text(s):
  # [FUNCTION DOC]
  # HELPER: converts bytes to text. Input: a byte string or any other value. Byte strings are
  # decoded as UTF-8; anything else is returned unchanged.
  if isinstance(s, bytes):
    return s.decode('utf-8')
  return s


class Client:
  # [CLASS DOC]
  # Client for the Cloud TPU API used to resolve a TPU name to network endpoints and to query
  # its state, health, symptoms, runtime version and accelerator type. It stores the TPU name,
  # whether the API is used (false when the name is a direct grpc address), an optional service
  # object, credentials, project, zone and discovery URL. Created with a name or from
  # environment variables; usable as a context manager, which does nothing on entry or exit.
  """Client for working with the Cloud TPU API.

  This client is intended to be used for resolving tpu name to ip addresses.

  It's recommended to use this library as a contextlib to utilize all
  functionality.
  """

  def __init__(self,
               tpu=None,
               zone=None,
               project=None,
               credentials='default',
               service=None,
               discovery_url=None):
    # [FUNCTION DOC]
    # Creates a client. Inputs (all optional): tpu (name, grpc address, or a single-item list;
    # falls back to environment variables and the TPU config), zone, project (auto-detected from
    # GCE metadata when missing and the API is used; also may come from the TPU config),
    # credentials (default means application default credentials), service (prebuilt API object
    # to use instead of building one), discovery_url (API discovery override; the environment
    # variable takes precedence). Raises a value error for an empty list or when no name can be
    # found, and a not-implemented error for a list of more than one TPU. Side effect: may call
    # the GCE metadata server.
    if isinstance(tpu, list):
      if not tpu:
        raise ValueError('At least one TPU must be specified.')
      if len(tpu) != 1:
        raise NotImplementedError(
            'Using multiple TPUs in a single session is not yet implemented')
      tpu = tpu[0]

    tpu = _get_tpu_name(tpu)

    if tpu is None:
      tpu_node_config = _get_tpu_node_config()
      if tpu_node_config:
        tpu = tpu_node_config.get('tpu_node_name')
        project = project or tpu_node_config.get('project')
        zone = zone or tpu_node_config.get('zone')
      else:
        raise ValueError('Please provide a TPU Name to connect to.')

    self._tpu = _as_text(tpu)

    self._use_api = not self._tpu.startswith('grpc://')
    self._service = service

    self._credentials = None
    self._project = None
    self._zone = None
    self._discovery_url = None
    if self._use_api:
      if credentials != 'default':
        self._credentials = credentials
      # Automatically detect project and zone if unspecified.
      if project:
        self._project = _as_text(project)
      else:
        self._project = _request_compute_metadata('project/project-id')
      if zone:
        self._zone = _as_text(zone)
      else:
        zone_path = _request_compute_metadata('instance/zone')
        self._zone = zone_path.split('/')[-1]
      self._discovery_url = _environment_discovery_url() or discovery_url

  def _symptom_msg(self, msg):
    # [FUNCTION DOC]
    # HELPER: formats log messages about TPU symptoms. Input: message text. Returns it prefixed
    # with the symptom label.
    """Return the structured Symptom message."""
    return 'Symptom: ' + msg

  def _oom_event(self, symptoms):
    # [FUNCTION DOC]
    # HELPER: supports the recoverable check. Input: symptoms, a list of symptom records (may be
    # empty or nothing). Scans from newest to oldest for out-of-memory records and returns true,
    # after logging a warning, if one was created within the last ninety seconds; otherwise false.
    # Creation times are parsed as UTC timestamps without fractional seconds.
    """Check if a runtime OOM event is reported."""
    if not symptoms:
      return False
    for symptom in reversed(symptoms):
      if symptom['symptomType'] != 'OUT_OF_MEMORY':
        continue
      oom_datetime_str = symptom['createTime'].split('.')[0]
      oom_datetime = datetime.datetime.strptime(oom_datetime_str,
                                                '%Y-%m-%dT%H:%M:%S')
      time_diff = _utcnow() - oom_datetime
      if time_diff < datetime.timedelta(seconds=_OOM_EVENT_COOL_TIME_SEC):
        logging.warning(
            self._symptom_msg(
                'a recent runtime OOM has occurred ~{} seconds ago. The model '
                'script will terminate automatically. To prevent future OOM '
                'events, please consider reducing the model size. To disable this '
                'behavior, set flag --runtime_oom_exit=false when starting the '
                'script.'.format(time_diff.seconds)))
        return True
    return False

  def _hbm_oom_event(self, symptoms):
    # [FUNCTION DOC]
    # HELPER: supports the recoverable check. Same as the runtime out-of-memory check but for
    # high-bandwidth-memory out-of-memory symptoms. Input: list of symptom records (may be empty
    # or nothing). Returns true, after logging a warning, if one was created within the last
    # ninety seconds; otherwise false.
    """Check if a HBM OOM event is reported."""
    if not symptoms:
      return False
    for symptom in reversed(symptoms):
      if symptom['symptomType'] != 'HBM_OUT_OF_MEMORY':
        continue
      oom_datetime_str = symptom['createTime'].split('.')[0]
      oom_datetime = datetime.datetime.strptime(oom_datetime_str,
                                                '%Y-%m-%dT%H:%M:%S')
      time_diff = _utcnow() - oom_datetime
      if time_diff < datetime.timedelta(seconds=_OOM_EVENT_COOL_TIME_SEC):
        logging.warning(
            self._symptom_msg(
                'a recent HBM OOM has occurred ~{} seconds ago. The model '
                'script will terminate automatically. To prevent future HBM OOM '
                'events, please consider reducing the model size. To disable this '
                'behavior, set flag --hbm_oom_exit=false when starting the '
                'script.'.format(time_diff.seconds)))
        return True
    return False

  def _tpu_service(self):
    # [FUNCTION DOC]
    # HELPER: obtains the Cloud TPU API object for other methods.
    # Returns the user-supplied service if one was given; otherwise builds a fresh version one
    # TPU API object (using application default credentials unless credentials were given, and
    # the discovery URL if configured, with discovery caching off). Raises a runtime error if the
    # Google API client libraries are not installed. Side effect: may contact credential and
    # discovery services.
    """Creates a new Cloud TPU API object.

    This works around an issue where the underlying HTTP connection sometimes
    times out when the script has been running for too long. Other methods in
    this object call this method to get a new API object whenever they need
    to communicate with the Cloud API.

    Raises:
      RuntimeError: If the dependent Python packages are missing.

    Returns:
      A Google Cloud TPU API object.
    """
    if self._service:
      return self._service

    if not _GOOGLE_API_CLIENT_INSTALLED:
      raise RuntimeError('Missing runtime dependency on the Google API client. '
                         'Run `pip install cloud-tpu-client` to fix.')

    credentials = self._credentials
    if credentials is None or credentials == 'default':
      credentials = client.GoogleCredentials.get_application_default()

    if self._discovery_url:
      return discovery.build(
          'tpu',
          'v1',
          credentials=credentials,
          discoveryServiceUrl=self._discovery_url,
          cache_discovery=False)
    else:
      return discovery.build(
          'tpu', 'v1', credentials=credentials, cache_discovery=False)

  def _full_name(self):
    # [FUNCTION DOC]
    # HELPER: builds the fully qualified API resource name of the TPU node from the project,
    # zone and TPU name. Returns the string; no side effects.
    """Returns the full Cloud name for this TPU."""
    return 'projects/%s/locations/%s/nodes/%s' % (
        self._project, self._zone, self._tpu)

  def _fetch_cloud_tpu_metadata(self):
    # [FUNCTION DOC]
    # HELPER: retrieves the TPU node description from the Cloud TPU API for the other getters.
    # Returns the metadata mapping. Any exception from building the service or executing the
    # request is not distinguished: request failures are converted to a value error mentioning
    # the TPU name (the original error text is included). Side effect: network access.
    """Returns the TPU metadata object from the TPU Get API call."""
    service = self._tpu_service()
    try:
      r = service.projects().locations().nodes().get(name=self._full_name())
      return r.execute()
    except Exception as e:
      raise ValueError("Could not lookup TPU metadata from name '%s'. Please "
                       'doublecheck the tpu argument in the TPUClusterResolver '
                       'constructor. Exception: %s' % (self._tpu, e))

  def _get_tpu_property(self, key):
    # [FUNCTION DOC]
    # HELPER: shared accessor for single TPU metadata fields. Input: key, the field name.
    # When the API is in use, fetches the metadata and returns the field (nothing if missing);
    # for direct grpc addresses returns nothing without any network access.
    if self._use_api:
      metadata = self._fetch_cloud_tpu_metadata()
      return metadata.get(key)

    return None

  def __enter__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: returns nothing, so binding the context manager to a name yields nothing
    # rather than the client; the class docs recommend context manager use but it adds nothing.
    # Context manager entry: only sets an internal open flag that is never read.
    self._open = True

  def __exit__(self, type, value, traceback):  # pylint: disable=redefined-builtin
    # [FUNCTION DOC]
    # Context manager exit: ignores the exception information and does nothing; exceptions
    # raised inside the block are not suppressed.
    del type, value, traceback

  def recoverable(self):
    # [FUNCTION DOC]
    # Reports whether training can eventually resume on this TPU. Queries the state and the
    # symptoms (network calls when the API is used). Returns false if the state is terminated or
    # preempted, if the runtime-OOM exit flag is on and a recent runtime OOM symptom exists, or if
    # the HBM-OOM exit flag is on and a recent HBM OOM symptom exists; otherwise true. May log
    # warnings; may raise the value error from metadata lookup failures.
    """Returns true if the TPU is in a state where training should eventually resume.

    If false the TPU is in a unrecoverable state and should be recreated.
    """
    state = self.state()
    symptoms = self.symptoms()
    if state and state in ['TERMINATED', 'PREEMPTED']:
      return False
    elif FLAGS.runtime_oom_exit and self._oom_event(symptoms):
      return False
    elif FLAGS.hbm_oom_exit and self._hbm_oom_event(symptoms):
      return False
    return True

  def symptoms(self):
    # [FUNCTION DOC]
    # Returns the list of symptom records reported for the TPU, or nothing when none are present
    # or the API is not used. Network access when the API is in use.
    """Return Cloud TPU Symptoms of the TPU."""
    return self._get_tpu_property('symptoms')

  def state(self):
    # [FUNCTION DOC]
    # Returns the TPU state string (for example ready or preempted) from the API, or nothing when
    # the API is not used or the field is missing. Network access when the API is in use.
    """Return state of the TPU."""
    return self._get_tpu_property('state')

  def health(self):
    # [FUNCTION DOC]
    # Returns the TPU health string from the API, or nothing when the API is not used or the
    # field is missing. Network access when the API is in use.
    """Return health of the TPU."""
    return self._get_tpu_property('health')

  def runtime_version(self):
    # [FUNCTION DOC]
    # Returns the TPU software (TensorFlow) version. With the API, reads the version field of
    # the metadata. For direct grpc addresses, asks the version switcher service on the first
    # endpoint and returns its current version; returns nothing if that service answers not
    # found, and re-raises other HTTP errors. Network access in both cases.
    """Return runtime version of the TPU."""

    if not self._use_api:
      # Fallback on getting version directly from TPU.
      url = _VERSION_SWITCHER_ENDPOINT.format(
          self.network_endpoints()[0]['ipAddress'])
      try:
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req)
        version_details = json.loads(resp.read())
        return version_details.get('currentVersion')
      except urllib.error.HTTPError as e:
        status_code = e.code
        if status_code == 404:
          return None
        else:
          raise e
    return self._get_tpu_property('tensorflowVersion')

  def accelerator_type(self):
    # [FUNCTION DOC]
    # Returns the accelerator type string from the API metadata, or nothing when the API is not
    # used or the field is missing. Network access when the API is in use.
    """Return accelerator type of the TPU."""
    return self._get_tpu_property('acceleratorType')

  def api_available(self):
    # [FUNCTION DOC]
    # Returns true when the Cloud TPU API is used (the TPU was not given as a direct grpc
    # address), false otherwise. No side effects.
    """Return if the Cloud TPU API is available, if not certain features will not work."""
    return self._use_api

  def name(self):
    # [FUNCTION DOC]
    # Returns the TPU name as text (the grpc address itself if no name was given). No side effects.
    """Return the name of the tpu, or the ip address if name is not provided."""
    return self._tpu

  def get_local_ip(self):
    # [FUNCTION DOC]
    # Returns the internal IP address of the VM running this code, read from the GCE metadata
    # server. Network access; fails if not running on Google Cloud.
    """Return the local ip address of the Google Cloud VM the workload is running on."""
    return _request_compute_metadata('instance/network-interfaces/0/ip')

  def network_endpoints(self):
    # [FUNCTION DOC]
    # Returns a list of endpoint mappings (ip address and port). For direct grpc addresses they
    # are parsed from the name without network access. With the API, fetches metadata, raises a
    # runtime error unless the state is ready, and returns the network endpoints list if present,
    # else a single endpoint built from the top-level ip address and port fields.
    """Return a list of tpu endpoints."""
    if not self._use_api:
      return list(_environment_var_to_network_endpoints(self._tpu))
    response = self._fetch_cloud_tpu_metadata()

    if response.get('state') != 'READY':
      raise RuntimeError('TPU "%s" is not yet ready; state: "%s"' %
                         (self._tpu, response.get('state')))
    if 'networkEndpoints' in response:
      return response['networkEndpoints']
    else:
      return [{'ipAddress': response['ipAddress'], 'port': response['port']}]

  def wait_for_healthy(self, timeout_s=1200, interval=30):
    # [FUNCTION DOC]
    # KNOWN ISSUE: without the API the health is always nothing, so this never succeeds and ends
    # with the timeout error. Also the deadline is checked only before sleeping, so the wait may
    # end slightly before the full timeout.
    # Polls the health until it equals healthy. Inputs: timeout_s (seconds to wait, default 1200)
    # and interval (seconds between polls, default 30). Logs a warning on each poll and on
    # success, sleeps between polls, and raises a runtime error when the next poll would pass
    # the deadline. Returns nothing.
    """Wait for TPU to become healthy or raise error if timeout reached.

    Args:
      timeout_s (int): The timeout in seconds for waiting TPU to become healthy.
      interval (int): The interval in seconds to poll the TPU for health.

    Raises:
      RuntimeError: If the TPU doesn't become healthy by the timeout.
    """
    timeout = time.time() + timeout_s
    while self.health() != 'HEALTHY':
      logging.warning(
          ('Waiting for TPU "%s" with state "%s" '
           'and health "%s" to become healthy'),
          self.name(), self.state(), self.health())
      if time.time() + interval > timeout:
        raise RuntimeError(
            'Timed out waiting for TPU "%s" to become healthy' % self.name())
      time.sleep(interval)

    logging.warning('TPU "%s" is healthy.', self.name())

  def configure_tpu_version(self, version, restart_type='always'):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the result check at the end is dead code: the worker function returns nothing,
    # so the results are always empty-valued and the result method is never called; errors still
    # surface because iterating the executor results re-raises them. Also generic Exception
    # types are raised.
    # Switches the software version of every TPU worker. Inputs: version (software version
    # string) and restart_type (always by default, or only if needed). Gets the endpoints (see
    # that method for its errors), then in parallel sends a request to each worker's version
    # switcher service. Raises a generic exception when a worker answers not found (version not
    # available) or fails otherwise. Returns nothing. Side effect: reconfigures remote TPUs.
    """Configure TPU software version.

    Args:
      version (string): Version of software to configure the TPU with.
      restart_type (string): Restart behaviour when switching versions,
        defaults to always restart. Options are 'always', 'ifNeeded'.

    """

    def configure_worker(worker):
      # [FUNCTION DOC]
      # HELPER: per-worker task for the version switch, run in a thread pool.
      # Input: worker, an endpoint mapping with its ip address. Sends an empty-bodied request to
      # the worker's version switcher with the version and restart type. Returns nothing.
      # Raises a generic exception with a version-unavailable message on a not-found response and
      # a generic failure message on other HTTP errors.
      """Configure individual TPU worker.

      Args:
        worker: A dict with the field ipAddress where the configure request will
          be sent.
      """
      ip_address = worker['ipAddress']
      url = (_VERSION_SWITCHER_ENDPOINT + '/{}?restartType={}').format(
          ip_address, version, restart_type)
      req = urllib.request.Request(url, data=b'')
      try:
        urllib.request.urlopen(req)
      except urllib.error.HTTPError as e:
        status_code = e.code
        if status_code == 404:
          raise Exception(
              'Tensorflow version {} is not available on Cloud TPU, '
              'try a previous nightly version or refer to '
              'https://cloud.google.com/tpu/docs/release-notes for '
              'the latest official version.'.format(version))
        else:
          raise Exception('Failed to configure worker {}'.format(ip_address))

    workers = self.network_endpoints()

    with futures.ThreadPoolExecutor(max_workers=len(workers)) as executor:
      results = executor.map(configure_worker, workers)
      for result in results:
        if result:
          result.result()
