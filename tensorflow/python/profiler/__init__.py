# [PACKAGE DOC]
# Package for the legacy TensorFlow profiler (tfprof) Python API.
# This package file is intentionally empty of code: it exports nothing and has no import-time
# side effects. The public pieces live in submodules: model_analyzer (profiling and advice for
# graphs and run metadata), option_builder (fluent builder for profiling option mappings),
# and pprof_profiler (converting graphs and run metadata into pprof profile files). Related
# internal modules such as the tfprof logger supply op-log merging and trace recording.
