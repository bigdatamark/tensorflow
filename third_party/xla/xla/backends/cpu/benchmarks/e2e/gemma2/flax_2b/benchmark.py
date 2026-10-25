# [MODULE DOC]
# End-to-end CPU benchmark script for the Gemma 2 2B instruction-tuned model in the Flax/JAX
# implementation. At import time it reads the model directory from the MODEL_DIR environment
# variable, asserts that the tokenizer file and checkpoint exist, loads parameters, builds the
# transformer, tokenizer and a chat sampler with a 1024-token cache. When run as a script it
# measures, five times each, the latency of generating one token (time to first token) and of
# generating 128 tokens (end-to-end latency), prints mean and relative standard deviation for
# each, and derives the time per output token from the difference. Times are in milliseconds.
# Requires the gemma package.
# Copyright 2024 The OpenXLA Authors.
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
"""Benchmark gemma2-2b-it Flax performance."""

import datetime
import os
import statistics

from gemma import gm


GEMMA_VARIANT = 'gemma2-2b-it'

# Assign Gemma path
GEMMA_PATH = os.environ.get('MODEL_DIR')

# Ensure that the tokenizer is present
TOKENIZER_PATH = os.path.join(GEMMA_PATH, 'tokenizer.model')
assert os.path.isfile(TOKENIZER_PATH), 'Tokenizer not found!'

# Ensure that the checkpoint is present
CKPT_PATH = os.path.join(GEMMA_PATH, GEMMA_VARIANT)
assert os.path.exists(CKPT_PATH), 'Flax checkpoint not found!'

# Set up model sampler
params = gm.ckpts.load_params(CKPT_PATH)
transformer = gm.nn.Gemma2_2B()
tokenizer = gm.text.Gemma2Tokenizer(path=TOKENIZER_PATH)
sampler = gm.text.ChatSampler(
    model=transformer,
    tokenizer=tokenizer,
    params=params,
    cache_length=1024,
)

OUTPUT_TOKEN_LEN = 128
prompt = 'What is JAX in 3 bullet points?'


def benchmark_generation_time(output_token_len):
  # [FUNCTION DOC]
  # Input: maximum number of new tokens to generate (integer). Sends the fixed prompt to the
  # chat sampler, measuring wall-clock time around the call. Prints the reply only when the
  # requested length equals the full benchmark length. Returns the elapsed time in milliseconds
  # as a floating-point number. Uses the module-level sampler and prompt.
  """Benchmark generation time given output token length."""
  timestamp_start = datetime.datetime.now()
  reply = sampler.chat(prompt, max_new_tokens=output_token_len)
  timestamp_end = datetime.datetime.now()
  timer_delta = timestamp_end - timestamp_start
  # Prints generated tokens when benchmarking the full length.
  if output_token_len == OUTPUT_TOKEN_LEN:
    print(reply)
  return timer_delta.total_seconds() * 1000


def display_tpot():
  # [FUNCTION DOC]
  # KNOWN ISSUE: depends on module-level lists of measurements that exist only when the file is
  # run as a script; calling it from an importing module raises a name error.
  # Takes no arguments. Subtracts the mean time-to-first-token from the mean end-to-end latency
  # and divides by one less than the full output length to get the time per output token. Prints
  # it in milliseconds rounded to two decimals. Returns nothing.
  """Calculate the time per output token."""
  e2e_latency_mean = statistics.mean(latency_list)
  ttft_mean = statistics.mean(ttft_ms_list)
  generation_time_mean = e2e_latency_mean - ttft_mean
  tpot = generation_time_mean / (OUTPUT_TOKEN_LEN - 1)
  print(f'TPOT: {round(tpot, 2)} ms')


def display_benchmark_results(timer_list, metric_name):
  # [FUNCTION DOC]
  # Inputs: list of measurements in milliseconds (at least two, otherwise the standard deviation
  # computation raises a statistics error) and a metric label. Prints the label, the mean and the
  # sample standard deviation as a percentage of the mean. Returns nothing.
  """Display mean and stdev for a given metric."""
  mean_time = statistics.mean(timer_list)
  stdev_time = statistics.stdev(timer_list)
  stdev_time_percentage = (stdev_time / mean_time) * 100

  print(
      '%s: %.2f ms ± %.2f%%' % (metric_name, mean_time, stdev_time_percentage)
  )


if __name__ == '__main__':
  # Measure time to first token.
  ttft_ms_list = [benchmark_generation_time(1) for _ in range(5)]
  # Measure time for full tokens.
  latency_list = [benchmark_generation_time(OUTPUT_TOKEN_LEN) for _ in range(5)]

  # Display benchmark results
  display_benchmark_results(ttft_ms_list, 'TTFT')
  display_benchmark_results(latency_list, 'E2E Latency')
  display_tpot()
  del sampler
