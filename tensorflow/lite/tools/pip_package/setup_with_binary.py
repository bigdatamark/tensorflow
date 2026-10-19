# [MODULE DOC]
# Packaging script (setuptools) for the TensorFlow Lite pip package built from prebuilt binaries.
# Reads the package name and version from the environment variables for project name and package
# version (raises a key error if either is missing), derives short and long descriptions from
# this module's docstring, and calls the setup function with metadata, classifiers, all
# discovered packages, shared library and extension files as package data, and a numpy
# requirement. Marks the distribution as having extension modules. Import-time effect: running
# it performs the packaging command given on the command line.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow Lite is for mobile and embedded devices.

TensorFlow Lite is the official solution for running machine learning models on
mobile and embedded devices. It enables on-device machine learning inference
with low latency and a small binary size on Android, iOS, and other operating
systems.
"""

import os

from setuptools import find_packages
from setuptools import setup
PACKAGE_NAME = os.environ['PROJECT_NAME']
PACKAGE_VERSION = os.environ['PACKAGE_VERSION']
DOCLINES = __doc__.split('\n')

setup(
    name=PACKAGE_NAME.replace('_', '-'),
    version=PACKAGE_VERSION,
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    url='https://www.tensorflow.org/lite/',
    author='Google, LLC',
    author_email='packages@tensorflow.org',
    license='Apache 2.0',
    include_package_data=True,
    has_ext_modules=lambda: True,
    keywords='tflite tensorflow tensor machine learning',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=[]),
    package_dir={'': '.'},
    package_data={'': ['*.so', '*.pyd']},
    install_requires=[
        'numpy >= 1.23.2',  # Better to keep sync with both TF ci_build
        # and OpenCV-Python requirement.
    ],
)
