# [MODULE DOC]
# HELPER: packaging script for the standalone cloud-tpu-client pip distribution.
# Importing or running it immediately calls the setuptools setup routine with package metadata
# (name, description, classifiers, Apache license, keywords) and pinned dependencies on the Google
# API client and oauth2client. The version is read from the cloud_tpu_client version module with
# dashes removed. It has no functions; running it as part of a build creates distribution artifacts.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Cloud TPU Client package."""

from cloud_tpu_client.version import __version__
from setuptools import find_packages
from setuptools import setup

setup(
    name='cloud-tpu-client',
    version=__version__.replace('-', ''),
    description='Client for using Cloud TPUs',
    long_description='Client for using Cloud TPUs',
    url='https://cloud.google.com/tpu/',
    author='Google Inc.',
    author_email='packages@tensorflow.org',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='tensorflow tpu',
    install_requires=['google-api-python-client==1.8.0', 'oauth2client']
)
