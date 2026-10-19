# [MODULE DOC]
# Smoothing of streaming keyword-spotting results. Provides a simple result holder
# (RecognizeResult) and a sliding-window averager (RecognizeCommands) that turns per-clip
# class confidences into a stable detected command with suppression of repeats.
# Used by the streaming accuracy test tool in this package. No import-time side effects.
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
# ==============================================================================
"""Stream accuracy recognize commands."""
import collections

import numpy as np


class RecognizeResult(object):
  # [CLASS DOC]
  # Mutable holder for one recognition outcome: the detected command word, its confidence score
  # and whether it counts as a new command compared with the previous one. Exposed through
  # read/write properties backed by private fields. Created by callers and filled in by
  # RecognizeCommands.
  """Save recognition result temporarily.

  Attributes:
    founded_command: A string indicating the word just founded. Default value
      is '_silence_'
    score: A float representing the confidence of founded word. Default
      value is zero.
    is_new_command: A boolean indicating if the founded command is a new one
      against the last one. Default value is False.
  """

  def __init__(self,
               founded_command="_silence_",
               score=0.0,
               is_new_command=False):
    # [FUNCTION DOC]
    # Constructor. Stores the detected command (string, default "_silence_"), its score (float,
    # default 0.0) and the new-command flag (boolean, default false). All optional.
    # Returns nothing.
    """Construct a recognition result.

    Args:
      founded_command: A string indicating the word just founded.
      score: A float representing the confidence of founded word.
      is_new_command: A boolean indicating if the founded command is a new one
        against the last one.
    """
    self._founded_command = founded_command
    self._score = score
    self._is_new_command = is_new_command

  @property
  def founded_command(self):
    # [FUNCTION DOC]
    # Property getter. Returns the stored detected command string.
    return self._founded_command

  @founded_command.setter
  def founded_command(self, value):
    # [FUNCTION DOC]
    # Property setter. Replaces the stored detected command string with the given value.
    # Returns nothing.
    self._founded_command = value

  @property
  def score(self):
    # [FUNCTION DOC]
    # Property getter. Returns the stored confidence score (float).
    return self._score

  @score.setter
  def score(self, value):
    # [FUNCTION DOC]
    # Property setter. Replaces the stored confidence score with the given value.
    # Returns nothing.
    self._score = value

  @property
  def is_new_command(self):
    # [FUNCTION DOC]
    # Property getter. Returns the stored boolean flag saying whether the command is new.
    return self._is_new_command

  @is_new_command.setter
  def is_new_command(self, value):
    # [FUNCTION DOC]
    # Property setter. Replaces the stored new-command flag with the given value.
    # Returns nothing.
    self._is_new_command = value


class RecognizeCommands(object):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring lists attributes with names such as _label and
  # _average_window_duration, but the constructor actually stores _labels and
  # _average_window_duration_ms. It also says results are removed from the "head" while new ones
  # are appended at the tail.
  # Sliding-window smoother for streaming classifier output. Holds the label list, window length
  # in milliseconds, detection threshold, minimum suppression interval between commands, minimum
  # number of results needed, a queue of (timestamp, confidences) pairs, the label count, and the
  # last accepted label and its time. Create one per audio stream and feed results in time order
  # via process_latest_result.
  """Smooth the inference results by using average window.

  Maintain a slide window over the audio stream, which adds new result(a pair of
  the 1.confidences of all classes and 2.the start timestamp of input audio
  clip) directly the inference produces one and removes the most previous one
  and other abnormal values. Then it smooth the results in the window to get
  the most reliable command in this period.

  Attributes:
    _label: A list containing commands at corresponding lines.
    _average_window_duration: The length of average window.
    _detection_threshold: A confidence threshold for filtering out unreliable
      command.
    _suppression_ms: Milliseconds every two reliable founded commands should
      apart.
    _minimum_count: An integer count indicating the minimum results the average
      window should cover.
    _previous_results: A deque to store previous results.
    _label_count: The length of label list.
    _previous_top_label: Last founded command. Initial value is '_silence_'.
    _previous_top_time: The timestamp of _previous results. Default is -np.inf.
  """

  def __init__(self, labels, average_window_duration_ms, detection_threshold,
               suppression_ms, minimum_count):
    # [FUNCTION DOC]
    # Constructor. Takes the list of label strings, the averaging window length in ms, the
    # confidence threshold a top score must exceed, the suppression interval in ms between two
    # accepted commands, and the minimum number of results required in the window. Initializes an
    # empty result queue, the label count, the previous top label to "_silence_" and the previous
    # top time to negative infinity. Returns nothing.
    """Init the RecognizeCommands with parameters used for smoothing."""
    # Configuration
    self._labels = labels
    self._average_window_duration_ms = average_window_duration_ms
    self._detection_threshold = detection_threshold
    self._suppression_ms = suppression_ms
    self._minimum_count = minimum_count
    # Working Variable
    self._previous_results = collections.deque()
    self._label_count = len(labels)
    self._previous_top_label = "_silence_"
    self._previous_top_time = -np.inf

  def process_latest_result(self, latest_results, current_time_ms,
                            recognize_element):
    # [FUNCTION DOC]
    # Adds a new inference result and fills the given result holder with the smoothed outcome.
    # Inputs: latest_results (array of per-label confidences, length must equal the label count),
    # current_time_ms (timestamp of the clip start, must not be earlier than the newest stored
    # one), recognize_element (a RecognizeResult that is mutated).
    # Raises a value error if the array length mismatches the label count or if the timestamp is
    # earlier than the previously stored one (compared with the most recently appended entry).
    # Behavior: appends the result, drops entries older than the window, and if too few entries
    # (below the minimum count) or the covered duration is under a quarter of the window, sets the
    # holder to the previous top label, score zero, not new, and returns. Otherwise averages the
    # scores per label, picks the best label, and marks it as new only if its average exceeds the
    # threshold, differs from the previous top label, and enough time (suppression) has passed
    # since the last accepted command. A new command updates the stored previous label and time.
    # The holder always receives the current best label and its average score in that case.
    # Returns nothing.
    """Smoothing the results in average window when a new result is added in.

    Receive a new result from inference and put the founded command into
    a RecognizeResult instance after the smoothing procedure.

    Args:
      latest_results: A list containing the confidences of all labels.
      current_time_ms: The start timestamp of the input audio clip.
      recognize_element: An instance of RecognizeResult to store founded
        command, its scores and if it is a new command.

    Raises:
      ValueError: The length of this result from inference doesn't match
        label count.
      ValueError: The timestamp of this result is earlier than the most
        previous one in the average window
    """
    if latest_results.shape[0] != self._label_count:
      raise ValueError("The results for recognition should contain {} "
                       "elements, but there are {} produced".format(
                           self._label_count, latest_results.shape[0]))
    if (self._previous_results.__len__() != 0 and
        current_time_ms < self._previous_results[0][0]):
      raise ValueError("Results must be fed in increasing time order, "
                       "but receive a timestamp of {}, which was earlier "
                       "than the previous one of {}".format(
                           current_time_ms, self._previous_results[0][0]))

    # Add the latest result to the head of the deque.
    self._previous_results.append([current_time_ms, latest_results])

    # Prune any earlier results that are too old for the averaging window.
    time_limit = current_time_ms - self._average_window_duration_ms
    while time_limit > self._previous_results[0][0]:
      self._previous_results.popleft()

    # If there are too few results, the result will be unreliable and bail.
    how_many_results = self._previous_results.__len__()
    earliest_time = self._previous_results[0][0]
    sample_duration = current_time_ms - earliest_time
    if (how_many_results < self._minimum_count or
        sample_duration < self._average_window_duration_ms / 4):
      recognize_element.founded_command = self._previous_top_label
      recognize_element.score = 0.0
      recognize_element.is_new_command = False
      return

    # Calculate the average score across all the results in the window.
    average_scores = np.zeros(self._label_count)
    for item in self._previous_results:
      score = item[1]
      for i in range(score.size):
        average_scores[i] += score[i] / how_many_results

    # Sort the averaged results in descending score order.
    sorted_averaged_index_score = []
    for i in range(self._label_count):
      sorted_averaged_index_score.append([i, average_scores[i]])
    sorted_averaged_index_score = sorted(
        sorted_averaged_index_score, key=lambda p: p[1], reverse=True)

    # Use the information of previous result to get current result
    current_top_index = sorted_averaged_index_score[0][0]
    current_top_label = self._labels[current_top_index]
    current_top_score = sorted_averaged_index_score[0][1]
    time_since_last_top = 0
    if (self._previous_top_label == "_silence_" or
        self._previous_top_time == -np.inf):
      time_since_last_top = np.inf
    else:
      time_since_last_top = current_time_ms - self._previous_top_time
    if (current_top_score > self._detection_threshold and
        current_top_label != self._previous_top_label and
        time_since_last_top > self._suppression_ms):
      self._previous_top_label = current_top_label
      self._previous_top_time = current_time_ms
      recognize_element.is_new_command = True
    else:
      recognize_element.is_new_command = False
    recognize_element.founded_command = current_top_label
    recognize_element.score = current_top_score
