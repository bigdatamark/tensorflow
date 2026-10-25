# [MODULE DOC]
# End-to-end CPU benchmark script for the Gemma 2 2B model using the Keras NLP implementation.
# Loads the model and tokenizer from the named preset, then for output limits of one token and
# thirty tokens runs one warm-up generation plus five timed generations of a fixed query,
# discarding statistical outliers, and reports mean time with a relative range. From the two
# results it prints time to first token and time per output token. Module constants control the
# output length, the query and verbosity. No import-time side effects other than imports; the
# benchmark runs only when executed as a script. Requires keras_nlp and numpy.
# Copyright 2025 The OpenXLA Authors.
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
"""Benchmark Gemma2-2B Keras performance."""

import time
import keras_nlp
import numpy as np

_NUM_OUTPUT_TOKENS = 30
_QUERY = "What is JAX in 3 bullet points?"
_VERBOSE = True


def compute_stats(array):
  # [FUNCTION DOC]
  # Input: array of numeric timings. Computes the first and third quartiles, discards values
  # outside 1.5 times the interquartile range beyond them, then returns a pair: the mean of the
  # remaining values and the largest deviation from that mean (to the min or max) as a percentage
  # of the mean. Pure function (uses numpy).
  """Reports mean and ± range for the given array.

  The range computation follows benchstat's.

  Args:
    array: The array to compute stats for.

  Returns:
    mean and ± %diff range.
  """
  q1 = np.percentile(array, 25)
  q3 = np.percentile(array, 75)
  low = q1 - 1.5 * (q3 - q1)
  high = q3 + 1.5 * (q3 - q1)

  # Remove outliers.
  filtered_array = list(filter(lambda x: low <= x and x <= high, array))

  mean = np.mean(filtered_array)
  min_val = np.min(filtered_array)
  max_val = np.max(filtered_array)
  max_diff = max(max_val - mean, mean - min_val)
  diff = max_diff / mean * 100.0

  return (mean, diff)


def run(gemma_lm, tokenizer, max_len):
  # [FUNCTION DOC]
  # KNOWN ISSUE: when max_len is below one it prints an error and calls the interactive exit
  # helper, terminating the whole process instead of raising an error. The docstring promises the
  # actual number of output tokens but the value returned is the count measured on the warm-up run
  # only, and the later runs only assert the count is equal.
  # Inputs: the Keras Gemma causal language model, its tokenizer and the maximum number of new
  # tokens. Performs one warm-up generation and five timed generations of the module query with
  # the total length limit (input tokens plus max_len, plus one), asserting each timed run
  # produces the same token count as the warm-up. Prints progress when verbose. Returns a tuple
  # of mean time in milliseconds, relative range percentage and the number of generated tokens.
  """Benchmarks inferences with at most `max_len` output tokens.

  Args:
    gemma_lm: The Gemma2 Keras model.
    tokenizer: The tokenizer for the Gemma2 model.
    max_len: The maximum number of output tokens per one inference.

  Returns:
    mean ± %diff and the actual number of output tokens generated per inference.
  """
  in_tokens = tokenizer(_QUERY)
  in_tokens_len = len(in_tokens)
  total_output_len = in_tokens_len + max_len
  if max_len < 1:
    print(f"Error: max_len {max_len} should be >= 1")
    exit()

  # Warm up.
  start = time.time()
  output = gemma_lm.generate(_QUERY, max_length=total_output_len + 1)
  warmup_time = (time.time() - start) * 1000
  num_actual_output_tokens = len(tokenizer(output))
  gen_tokens = max(num_actual_output_tokens - in_tokens_len, 0)

  print("Warmup: Number of generated output tokens: ", gen_tokens)

  if _VERBOSE:
    print("=== Max len: %d ===" % max_len)
    print("Warmup: %lf ms" % warmup_time)
    print("Output:\n%s\n" % output)

  times = []
  for i in range(1, 6):
    start = time.time()
    output = gemma_lm.generate(_QUERY, max_length=total_output_len + 1)
    elapsed_time = (time.time() - start) * 1000
    assert num_actual_output_tokens == len(tokenizer(output))
    times.append(elapsed_time)

    if _VERBOSE:
      print("%d: %lf ms" % (i, elapsed_time))
      print("Benchmark: Number of generated output tokens: ", gen_tokens)

  mean, diff = compute_stats(times)
  if _VERBOSE:
    print("Mean: %lf ± %d%% ms\n" % (mean, diff))

  return (mean, diff, gen_tokens)


def main():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the line announcing the generated token count passes the number as a second
  # print argument to a format string, so the placeholder is printed literally instead of being
  # filled. The time-per-token range printed uses the range of the long run only.
  # Loads the model and tokenizer from the preset, runs the benchmark for one output token and
  # for the configured longer length, then prints the time to first token and the time per output
  # token (difference of the means divided by the extra tokens). Takes no arguments, returns
  # nothing.
  if _VERBOSE:
    print("Query: %s" % _QUERY)

  model_name = "gemma2_2b_en"
  gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_name)
  tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_name)
  mean_1, diff_1, _ = run(gemma_lm, tokenizer, 1)
  mean_n, diff_n, num_output_tokens = run(
      gemma_lm, tokenizer, _NUM_OUTPUT_TOKENS
  )

  print("Generated %d tokens", num_output_tokens)
  tpot = (mean_n - mean_1) / (num_output_tokens - 1)
  print("TTFT: %lf ± %d%% ms" % (mean_1, diff_1))
  print("TPOT: %lf ± %d%% ms" % (tpot, diff_n))


if __name__ == "__main__":
  main()
