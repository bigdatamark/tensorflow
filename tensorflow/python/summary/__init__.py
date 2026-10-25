# [PACKAGE DOC]
# Package for the internal TensorFlow summary machinery (the Python side of TensorBoard logging).
# This initializer is empty: it exports nothing and has no import-time side effects, so each
# submodule must be imported directly. Submodules: summary (the version 1 summary-producing
# operations and file-writer helpers), plugin_asset (graph-scoped singleton assets that
# TensorBoard plugins can attach), summary_iterator (reading event records back from event
# files), tb_summary (thin re-exports of the version 2 summary API that is provided by the
# TensorBoard package), and the test modules that exercise them.
