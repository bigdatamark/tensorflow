# [MODULE DOC]
# Executable script that installs the TPU GRPC worker as a per-user systemd service. At import
# time it computes the current user name and the Python search path and builds the text of a
# systemd unit file (module constants) that launches the start_grpc_tpu_worker command. When run
# as a script it writes the unit file, moves it to the user's systemd directory, then reloads,
# enables and starts the service via systemctl. Exits with status 1 if the unit already exists.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-import-not-at-top
"""Script to start GRPC worker as a service.

Usage:
    python3 grpc_tpu_worker_service.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_sys_path():
  # [FUNCTION DOC]
  # HELPER: supplies the Python path for the generated service file.
  # Returns the interpreter's module search path entries joined into one colon-separated string.
  return ":".join(sys.path)


def get_username():
  # [FUNCTION DOC]
  # HELPER: supplies the user name for the generated service file.
  # Returns the value of the USER environment variable, or the null value if it is not set (in
  # which case the generated path would contain the text for null).
  return os.environ.get("USER")


username = get_username()
sys_path = get_sys_path()

SERVICE_FILE_CONTENT = f"""
[Unit]
Description=GRPC TPU Worker Service
After=network.target
[Service]
Type=simple
Environment="PYTHONPATH=$PYTHONPATH{sys_path}"
EnvironmentFile=/home/tpu-runtime/tpu-env
#ExecStartPre=/bin/mkdir -p /tmp/tflogs
#ExecStartPre=/bin/touch /tmp/tflogs/grpc_tpu_worker.log
#ExecStartPre=/bin/chmod +r /tmp/tflogs
ExecStart=/home/{get_username()}/.local/bin/start_grpc_tpu_worker #2>&1 | tee -a /tmp/tflogs/grpc_tpu_worker.log
Restart=on-failure
# Restart service after 10 seconds if the service crashes:
RestartSec=10
[Install]
WantedBy=multi-user.target
"""
SERVICE_NAME = "grpc_tpu_worker.service"


def create_systemd_service_file(service_content, service_name):
  # [FUNCTION DOC]
  # HELPER: first step of the service installation.
  # Writes the given text content to a file named service_name in the current working directory
  # (overwriting any existing file) and prints a confirmation. Returns nothing; raises an OS error
  # if the file cannot be written.
  with open(service_name, "w") as file:
    file.write(service_content)
  print(f"Service file {service_name} created")


def move_file_to_systemd(service_name):
  # [FUNCTION DOC]
  # HELPER: second step of the service installation.
  # Ensures the per-user systemd directory under the home config folder exists (creating parents
  # and printing a message if it had to be created), then moves the file named service_name from
  # the current directory into it and prints the destination. Returns nothing; may raise OS or
  # file-move errors.
  user_systemd_dir = Path.home() / ".config" / "systemd" / "user"
  if not user_systemd_dir.exists():
    user_systemd_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory {user_systemd_dir}")

  source_file = Path(service_name)
  dest_file = user_systemd_dir / service_name
  shutil.move(str(source_file), str(dest_file))
  print(f"Service file moved to {dest_file}")


def enable_start_service(service_name):
  # [FUNCTION DOC]
  # HELPER: final step of the service installation.
  # Runs, in order, the user-level systemctl commands to import the environment, reload the
  # daemon, enable the named service and start it, printing each executed command. A failing
  # command raises a called-process error (checking is enabled), stopping the sequence.
  commands = [
      ["systemctl", "--user", "import-environment"],
      ["systemctl", "--user", "daemon-reload"],
      ["systemctl", "--user", "enable", service_name],
      ["systemctl", "--user", "start", service_name],
  ]
  for command in commands:
    subprocess.run(command, check=True)
    print(f"Executed: {' '.join(command)}")


def run():
  # [FUNCTION DOC]
  # Script entry point. If the user-level unit file for the worker service already exists, prints
  # a message and exits the process with status 1. Otherwise creates the unit file from the
  # module-level content, moves it into the user systemd directory and enables and starts it.
  # Takes no inputs and returns nothing.
  service_file_path = (
      Path.home() / ".config" / "systemd" / "user" / SERVICE_NAME
  )
  if service_file_path.exists():
    print(f"Service file {service_file_path} already exists")
    sys.exit(1)
  else:
    create_systemd_service_file(SERVICE_FILE_CONTENT, SERVICE_NAME)
    move_file_to_systemd(SERVICE_NAME)
    enable_start_service(SERVICE_NAME)


if __name__ == "__main__":
  run()
