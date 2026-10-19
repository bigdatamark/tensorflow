# [MODULE DOC]
# Data loading and preprocessing for the speech commands example. Provides label constants,
# helpers to build the word list, assign files to train/validation/test partitions by a stable
# hash of the file name, read and write wav files, report expected feature ranges, and the
# AudioProcessor class that downloads the data set, indexes it, loads background noise, builds a
# graph to turn wavs into spectrogram/MFCC/micro-frontend features and serves batches.
# Import-time side effects: disables eager execution globally in TensorFlow, and tries to import
# the optional micro frontend op (falls back to a null value if unavailable).
# The docstring at the top of the file says "model definitions" which is inaccurate; the module is
# about data handling.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Model definitions for simple speech recognition.

"""
import hashlib
import math
import os.path
import random
import re
import sys
import tarfile

import numpy as np
import urllib
import tensorflow as tf

from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

tf.compat.v1.disable_eager_execution()

# If it's available, load the specialized feature generator. If this doesn't
# work, try building with bazel instead of running the Python script directly.
try:
  from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op  # pylint:disable=g-import-not-at-top
except ImportError:
  frontend_op = None

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185


def prepare_words_list(wanted_words):
  # [FUNCTION DOC]
  # Returns a new list: the silence label, then the unknown label, then the given wanted words
  # (a list of strings) in order. Does not modify the input. Raises no errors of its own.
  """Prepends common tokens to the custom word list.

  Args:
    wanted_words: List of strings containing the custom words.

  Returns:
    List with the standard silence and unknown tokens added.
  """
  return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
  # [FUNCTION DOC]
  # Deterministically assigns a file to a partition. Takes the base name of the path, strips
  # everything from the grouping marker "_nohash_" onward, hashes the rest with SHA-1 and maps the
  # hash to a pseudo-percentage between 0 and 100. Values below validation_percentage give
  # "validation", values below validation plus testing percentage give "testing", otherwise
  # "training". Inputs: file path string and two numeric percentages. Returns one of those three
  # strings. No side effects, no exceptions expected for valid strings.
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


def load_wav_file(filename):
  # [FUNCTION DOC]
  # Reads a wav file from the given path using a throwaway graph and session, decoded to a single
  # channel, and returns the samples as a flat numpy array of floats in the range -1 to 1.
  # Errors from file reading or decoding propagate. No persistent state is changed.
  """Loads an audio file and returns a float PCM-encoded array of samples.

  Args:
    filename: Path to the .wav file to load.

  Returns:
    Numpy array holding the sample data as floats between -1.0 and 1.0.
  """
  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
    return sess.run(
        wav_decoder,
        feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def save_wav_file(filename, wav_data, sample_rate):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says wav_data is a 2D array, but any array is accepted because it
  # is reshaped to a single column of samples before encoding.
  # Encodes float samples as a mono wav and writes it to the given path, using a throwaway graph
  # and session. Inputs: output path, sample data array, integer sample rate. Side effect: creates
  # or overwrites the file. Returns nothing; file or encoding errors propagate.
  """Saves audio sample data to a .wav audio file.

  Args:
    filename: Path to save the file to.
    wav_data: 2D array of float PCM-encoded audio data.
    sample_rate: Samples per second to encode in the file.
  """
  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
    sample_rate_placeholder = tf.compat.v1.placeholder(tf.int32, [])
    wav_data_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 1])
    wav_encoder = tf.audio.encode_wav(wav_data_placeholder,
                                      sample_rate_placeholder)
    wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
    sess.run(
        wav_saver,
        feed_dict={
            wav_filename_placeholder: filename,
            sample_rate_placeholder: sample_rate,
            wav_data_placeholder: np.reshape(wav_data, (-1, 1))
        })


def get_features_range(model_settings):
  # [FUNCTION DOC]
  # Returns a pair (minimum, maximum) of expected feature values depending on the preprocessing
  # mode in the model settings mapping: average gives 0 to 127.5, mfcc gives -247 to 30, micro
  # gives 0 to 26. The values are empirical. Raises a generic exception for any other mode.
  """Returns the expected min/max for generated features.

  Args:
    model_settings: Information about the current model being trained.

  Returns:
    Min/max float pair holding the range of features.

  Raises:
    Exception: If preprocessing mode isn't recognized.
  """
  # TODO(petewarden): These values have been derived from the observed ranges
  # of spectrogram and MFCC inputs. If the preprocessing pipeline changes,
  # they may need to be updated.
  if model_settings['preprocess'] == 'average':
    features_min = 0.0
    features_max = 127.5
  elif model_settings['preprocess'] == 'mfcc':
    features_min = -247.0
    features_max = 30.0
  elif model_settings['preprocess'] == 'micro':
    features_min = 0.0
    features_max = 26.0
  else:
    raise Exception('Unknown preprocess mode "%s" (should be "mfcc",'
                    ' "average", or "micro")' % (model_settings['preprocess']))
  return features_min, features_max


class AudioProcessor(object):
  # [CLASS DOC]
  # Loads, partitions and serves speech command audio. State: data_dir, data_index (mapping from
  # partition name to list of entries with label and file), words_list, word_to_index,
  # background_data (list of long noise arrays), and placeholders/output tensors of the processing
  # graph (wav filename, foreground volume, time shift padding/offset, background data/volume,
  # output, merged summaries, optional summary writer). Created once per session; graph building
  # happens in the default graph, so it must be created in the graph later used with the session.
  """Handles loading, partitioning, and preparing audio training data."""

  def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
               wanted_words, validation_percentage, testing_percentage,
               model_settings, summaries_dir):
    # [FUNCTION DOC]
    # Constructor. If data_dir is non-empty: stores it, optionally downloads/extracts the data
    # (when data_url is non-empty), builds the partitioned file index (using silence and unknown
    # percentages, wanted words and validation/testing percentages) and loads background noise.
    # In all cases then builds the audio processing graph from model_settings and summaries_dir.
    # KNOWN ISSUE: when data_dir is empty, the data attributes (data_index, background_data, etc.)
    # are never set, so later data access fails.
    if data_dir:
      self.data_dir = data_dir
      self.maybe_download_and_extract_dataset(data_url, data_dir)
      self.prepare_data_index(silence_percentage, unknown_percentage,
                              wanted_words, validation_percentage,
                              testing_percentage)
      self.prepare_background_data()
    self.prepare_processing_graph(model_settings, summaries_dir)

  def maybe_download_and_extract_dataset(self, data_url, dest_directory):
    # [FUNCTION DOC]
    # Downloads and unpacks the data set archive. Does nothing if data_url is empty. Otherwise
    # creates dest_directory if missing and, only when the archive file named by the last URL
    # segment is not already there, downloads it with a progress display on standard output, logs
    # the size, and extracts the gzip tar into dest_directory. On download failure logs an error
    # and re-raises. Returns nothing.
    # KNOWN ISSUE: extraction happens only right after a fresh download; if the archive exists but
    # was never extracted, nothing is extracted. Also, the archive is extracted without checking
    # member paths.
    """Download and extract data set tar file.

    If the data set we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a
    directory.
    If the data_url is none, don't download anything and expect the data
    directory to contain the correct files already.

    Args:
      data_url: Web location of the tar file containing the data set.
      dest_directory: File path to extract data to.
    """
    if not data_url:
      return
    if not gfile.Exists(dest_directory):
      os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not gfile.Exists(filepath):

      def _progress(count, block_size, total_size):
        # [FUNCTION DOC]
        # HELPER: download progress callback for the dataset download.
        # Writes a carriage-return progress line with the file name and percentage downloaded
        # (from block count, block size and total size) to standard output and flushes it.
        # Returns nothing.
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' %
            (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      try:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
      except:
        tf.compat.v1.logging.error(
            'Failed to download URL: {0} to folder: {1}. Please make sure you '
            'have enough free space and an internet connection'.format(
                data_url, filepath))
        raise
      print()
      statinfo = os.stat(filepath)
      tf.compat.v1.logging.info(
          'Successfully downloaded {0} ({1} bytes)'.format(
              filename, statinfo.st_size))
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)

  def prepare_data_index(self, silence_percentage, unknown_percentage,
                         wanted_words, validation_percentage,
                         testing_percentage):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns a dictionary and lookup map, but it returns
    # nothing; results are stored in the data_index, words_list and word_to_index attributes.
    # Scans data_dir/*/*.wav, skipping the background noise folder. Each file's label is its
    # lower-cased folder name; it is assigned to a partition via which_set. Wanted words go to the
    # partition lists, others to a per-partition unknown pool. Seeds the global random generator
    # with a fixed seed for determinism. Raises a generic exception if no wavs are found or any
    # wanted word is missing from the folders. Then, per partition, appends silence entries
    # (ceil of silence_percentage of the wanted-word count, all pointing at the first training
    # file), shuffles the unknown pool and adds ceil of unknown_percentage of it, and shuffles each
    # partition. Finally builds the word list and the mapping from word to index (wanted words
    # start at 2, other words map to the unknown index, silence to 0).
    # Note: silence and unknown counts are based on the set size before silence was added.
    """Prepares a list of the samples organized by set and label.

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`, figures out the
    right
    labels for each file based on the name of the subdirectory it belongs to,
    and uses a stable hash to assign it to a data set partition.

    Args:
      silence_percentage: How much of the resulting data should be background.
      unknown_percentage: How much should be audio outside the wanted classes.
      wanted_words: Labels of the classes we want to be able to recognize.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.

    Raises:
      Exception: If expected files are not found.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index + 2
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}
    all_words = {}
    # Look through all the subfolders to find audio samples
    search_path = os.path.join(self.data_dir, '*', '*.wav')
    for wav_path in gfile.Glob(search_path):
      _, word = os.path.split(os.path.dirname(wav_path))
      word = word.lower()
      # Treat the '_background_noise_' folder as a special case, since we expect
      # it to contain long audio samples we mix in to improve training.
      if word == BACKGROUND_NOISE_DIR_NAME:
        continue
      all_words[word] = True
      set_index = which_set(wav_path, validation_percentage, testing_percentage)
      # If it's a known class, store its detail, otherwise add it to the list
      # we'll use to train the unknown label.
      if word in wanted_words_index:
        self.data_index[set_index].append({'label': word, 'file': wav_path})
      else:
        unknown_index[set_index].append({'label': word, 'file': wav_path})
    if not all_words:
      raise Exception('No .wavs found at ' + search_path)
    for index, wanted_word in enumerate(wanted_words):
      if wanted_word not in all_words:
        raise Exception('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))
    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    silence_wav_path = self.data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
      set_size = len(self.data_index[set_index])
      silence_size = int(math.ceil(set_size * silence_percentage / 100))
      for _ in range(silence_size):
        self.data_index[set_index].append({
            'label': SILENCE_LABEL,
            'file': silence_wav_path
        })
      # Pick some unknowns to add to each partition of the data set.
      random.shuffle(unknown_index[set_index])
      unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
      self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])
    # Prepare the rest of the result data structure.
    self.words_list = prepare_words_list(wanted_words)
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        self.word_to_index[word] = UNKNOWN_WORD_INDEX
    self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

  def prepare_background_data(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns the list of samples, but it only returns the
    # (empty) list when the background folder is missing; otherwise it returns nothing.
    # Sets the background_data attribute to a list of flat float arrays, one per wav in the
    # background noise folder under data_dir. If the folder does not exist the list stays empty
    # and no augmentation is done. Raises a generic exception if the folder exists but has no wavs.
    # Uses a throwaway graph and session for decoding.
    """Searches a folder for background noise audio, and loads it into memory.

    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.

    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.

    Returns:
      List of raw PCM-encoded audio samples of background noise.

    Raises:
      Exception: If files aren't found in the folder.
    """
    self.background_data = []
    background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
    if not gfile.Exists(background_dir):
      return self.background_data
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
      search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
                                 '*.wav')
      for wav_path in gfile.Glob(search_path):
        wav_data = sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
        self.background_data.append(wav_data)
      if not self.background_data:
        raise Exception('No background wav files were found in ' + search_path)

  def prepare_processing_graph(self, model_settings, summaries_dir):
    # [FUNCTION DOC]
    # Builds, in the default graph under a "data" name scope, the pipeline: read and decode a wav
    # (single channel, fixed desired sample count), scale by a foreground volume, pad and slice for
    # time shifting, add scaled background noise, clip to -1..1, compute a spectrogram, then reduce
    # it according to the preprocess mode: average pooling, MFCC, or micro frontend features
    # (scaled). Stores the input placeholders and the output tensor as attributes with a trailing
    # underscore, and image summaries for the stages. Merges summaries from the data scope and, if
    # summaries_dir is non-empty, creates a summary writer under its data subfolder.
    # Raises a generic exception if micro mode is requested but the frontend op is unavailable, and
    # a value error for an unknown preprocess mode. Uses the settings desired_samples, window sizes
    # and strides, fingerprint width, sample rate and average window width.
    # KNOWN ISSUE: the summary writer attribute only exists when summaries_dir is non-empty, but
    # get_data always uses it.
    """Builds a TensorFlow graph to apply the input distortions.

    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.

    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:

      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_padding_placeholder_: Where to pad the clip.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - output_: Output 2D fingerprint of processed audio.

    Args:
      model_settings: Information about the current model being trained.
      summaries_dir: Path to save training summary information to.

    Raises:
      ValueError: If the preprocessing mode isn't recognized.
      Exception: If the preprocessor wasn't compiled in.
    """
    with tf.compat.v1.get_default_graph().name_scope('data'):
      desired_samples = model_settings['desired_samples']
      self.wav_filename_placeholder_ = tf.compat.v1.placeholder(
          tf.string, [], name='wav_filename')
      wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
      wav_decoder = tf.audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)
      # Allow the audio sample's volume to be adjusted.
      self.foreground_volume_placeholder_ = tf.compat.v1.placeholder(
          tf.float32, [], name='foreground_volume')
      scaled_foreground = tf.multiply(wav_decoder.audio,
                                      self.foreground_volume_placeholder_)
      # Shift the sample's start position, and pad any gaps with zeros.
      self.time_shift_padding_placeholder_ = tf.compat.v1.placeholder(
          tf.int32, [2, 2], name='time_shift_padding')
      self.time_shift_offset_placeholder_ = tf.compat.v1.placeholder(
          tf.int32, [2], name='time_shift_offset')
      padded_foreground = tf.pad(
          tensor=scaled_foreground,
          paddings=self.time_shift_padding_placeholder_,
          mode='CONSTANT')
      sliced_foreground = tf.slice(padded_foreground,
                                   self.time_shift_offset_placeholder_,
                                   [desired_samples, -1])
      # Mix in background noise.
      self.background_data_placeholder_ = tf.compat.v1.placeholder(
          tf.float32, [desired_samples, 1], name='background_data')
      self.background_volume_placeholder_ = tf.compat.v1.placeholder(
          tf.float32, [], name='background_volume')
      background_mul = tf.multiply(self.background_data_placeholder_,
                                   self.background_volume_placeholder_)
      background_add = tf.add(background_mul, sliced_foreground)
      background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
      # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
      spectrogram = audio_ops.audio_spectrogram(
          background_clamp,
          window_size=model_settings['window_size_samples'],
          stride=model_settings['window_stride_samples'],
          magnitude_squared=True)
      tf.compat.v1.summary.image(
          'spectrogram', tf.expand_dims(spectrogram, -1), max_outputs=1)
      # The number of buckets in each FFT row in the spectrogram will depend on
      # how many input samples there are in each window. This can be quite
      # large, with a 160 sample window producing 127 buckets for example. We
      # don't need this level of detail for classification, so we often want to
      # shrink them down to produce a smaller result. That's what this section
      # implements. One method is to use average pooling to merge adjacent
      # buckets, but a more sophisticated approach is to apply the MFCC
      # algorithm to shrink the representation.
      if model_settings['preprocess'] == 'average':
        self.output_ = tf.nn.pool(
            input=tf.expand_dims(spectrogram, -1),
            window_shape=[1, model_settings['average_window_width']],
            strides=[1, model_settings['average_window_width']],
            pooling_type='AVG',
            padding='SAME')
        tf.compat.v1.summary.image('shrunk_spectrogram',
                                   self.output_,
                                   max_outputs=1)
      elif model_settings['preprocess'] == 'mfcc':
        self.output_ = audio_ops.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=model_settings['fingerprint_width'])
        tf.compat.v1.summary.image(
            'mfcc', tf.expand_dims(self.output_, -1), max_outputs=1)
      elif model_settings['preprocess'] == 'micro':
        if not frontend_op:
          raise Exception(
              'Micro frontend op is currently not available when running'
              ' TensorFlow directly from Python, you need to build and run'
              ' through Bazel')
        sample_rate = model_settings['sample_rate']
        window_size_ms = (model_settings['window_size_samples'] *
                          1000) / sample_rate
        window_step_ms = (model_settings['window_stride_samples'] *
                          1000) / sample_rate
        int16_input = tf.cast(tf.multiply(background_clamp, 32768), tf.int16)
        micro_frontend = frontend_op.audio_microfrontend(
            int16_input,
            sample_rate=sample_rate,
            window_size=window_size_ms,
            window_step=window_step_ms,
            num_channels=model_settings['fingerprint_width'],
            out_scale=1,
            out_type=tf.float32)
        self.output_ = tf.multiply(micro_frontend, (10.0 / 256.0))
        tf.compat.v1.summary.image(
            'micro',
            tf.expand_dims(tf.expand_dims(self.output_, -1), 0),
            max_outputs=1)
      else:
        raise ValueError('Unknown preprocess mode "%s" (should be "mfcc", '
                         ' "average", or "micro")' %
                         (model_settings['preprocess']))

      # Merge all the summaries and write them out to /tmp/retrain_logs (by
      # default)
      self.merged_summaries_ = tf.compat.v1.summary.merge_all(scope='data')
      if summaries_dir:
        self.summary_writer_ = tf.compat.v1.summary.FileWriter(
            summaries_dir + '/data', tf.compat.v1.get_default_graph())

  def set_size(self, mode):
    # [FUNCTION DOC]
    # Returns the number of entries in the partition named by mode (training, validation or
    # testing), including silence and unknown entries. Unknown mode raises a key error.
    """Calculates the number of samples in the dataset partition.

    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.

    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])

  def get_data(self, how_many, offset, model_settings, background_frequency,
               background_volume_range, time_shift, mode, sess):
    # [FUNCTION DOC]
    # Produces a batch of processed features and label indexes from a partition. how_many is the
    # batch size (-1 means the whole partition); offset is the start for deterministic reading.
    # In training mode samples are drawn at random with replacement; otherwise (or for -1) they are
    # taken in order from offset. Per sample: a random time shift in the allowed range is applied
    # when time_shift is positive; background noise is mixed in (random clip, volume random up to
    # background_volume_range with probability background_frequency) only in training mode when
    # background data exists; silence entries get full-range random background and zero
    # foreground. Runs the processing graph in the given session and writes summaries.
    # Returns a pair: array of flattened features (count by fingerprint size) and array of label
    # indexes. Raises a value error if a background clip is not longer than the desired samples.
    # KNOWN ISSUE: uses the summary writer even if none was created; silence entries with no
    # background data fail because the background list is empty; the time shift range excludes
    # the upper bound; an offset beyond the end with -1 is not validated.
    """Gather samples from the data set, applying transformations as needed.

    When the mode is 'training', a random selection of samples will be returned,
    otherwise the first N clips in the partition will be used. This ensures that
    validation always uses the same samples, reducing noise in the metrics.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      offset: Where to start when fetching deterministically.
      model_settings: Information about the current model being trained.
      background_frequency: How many clips will have background noise, 0.0 to
        1.0.
      background_volume_range: How loud the background noise will be.
      time_shift: How much to randomly shift the clips by in time.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.
      sess: TensorFlow session that was active when processor was created.

    Returns:
      List of sample data for the transformed samples, and list of label indexes

    Raises:
      ValueError: If background samples are too short.
    """
    # Pick one of the partitions to choose samples from.
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = max(0, min(how_many, len(candidates) - offset))
    # Data and labels will be populated and returned.
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    labels = np.zeros(sample_count)
    desired_samples = model_settings['desired_samples']
    use_background = self.background_data and (mode == 'training')
    pick_deterministically = (mode != 'training')
    # Use the processing graph we created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in range(offset, offset + sample_count):
      # Pick which audio sample to use.
      if how_many == -1 or pick_deterministically:
        sample_index = i
      else:
        sample_index = np.random.randint(len(candidates))
      sample = candidates[sample_index]
      # If we're time shifting, set up the offset for this sample.
      if time_shift > 0:
        time_shift_amount = np.random.randint(-time_shift, time_shift)
      else:
        time_shift_amount = 0
      if time_shift_amount > 0:
        time_shift_padding = [[time_shift_amount, 0], [0, 0]]
        time_shift_offset = [0, 0]
      else:
        time_shift_padding = [[0, -time_shift_amount], [0, 0]]
        time_shift_offset = [-time_shift_amount, 0]
      input_dict = {
          self.wav_filename_placeholder_: sample['file'],
          self.time_shift_padding_placeholder_: time_shift_padding,
          self.time_shift_offset_placeholder_: time_shift_offset,
      }
      # Choose a section of background noise to mix in.
      if use_background or sample['label'] == SILENCE_LABEL:
        background_index = np.random.randint(len(self.background_data))
        background_samples = self.background_data[background_index]
        if len(background_samples) <= model_settings['desired_samples']:
          raise ValueError(
              'Background sample is too short! Need more than %d'
              ' samples but only %d were found' %
              (model_settings['desired_samples'], len(background_samples)))
        background_offset = np.random.randint(
            0, len(background_samples) - model_settings['desired_samples'])
        background_clipped = background_samples[background_offset:(
            background_offset + desired_samples)]
        background_reshaped = background_clipped.reshape([desired_samples, 1])
        if sample['label'] == SILENCE_LABEL:
          background_volume = np.random.uniform(0, 1)
        elif np.random.uniform(0, 1) < background_frequency:
          background_volume = np.random.uniform(0, background_volume_range)
        else:
          background_volume = 0
      else:
        background_reshaped = np.zeros([desired_samples, 1])
        background_volume = 0
      input_dict[self.background_data_placeholder_] = background_reshaped
      input_dict[self.background_volume_placeholder_] = background_volume
      # If we want silence, mute out the main sample but leave the background.
      if sample['label'] == SILENCE_LABEL:
        input_dict[self.foreground_volume_placeholder_] = 0
      else:
        input_dict[self.foreground_volume_placeholder_] = 1
      # Run the graph to produce the output audio.
      summary, data_tensor = sess.run(
          [self.merged_summaries_, self.output_], feed_dict=input_dict)
      self.summary_writer_.add_summary(summary)
      data[i - offset, :] = data_tensor.flatten()
      label_index = self.word_to_index[sample['label']]
      labels[i - offset] = label_index
    return data, labels

  def get_features_for_wav(self, wav_filename, model_settings, sess):
    # [FUNCTION DOC]
    # Runs the processing graph on one wav file with no time shift, no background and full
    # foreground volume, using the given session. Returns the session result as a list holding
    # one array of features (not a bare array). Does not touch summaries.
    """Applies the feature transformation process to the input_wav.

    Runs the feature generation process (generally producing a spectrogram from
    the input samples) on the WAV file. This can be useful for testing and
    verifying implementations being run on other platforms.

    Args:
      wav_filename: The path to the input audio file.
      model_settings: Information about the current model being trained.
      sess: TensorFlow session that was active when processor was created.

    Returns:
      Numpy data array containing the generated features.
    """
    desired_samples = model_settings['desired_samples']
    input_dict = {
        self.wav_filename_placeholder_: wav_filename,
        self.time_shift_padding_placeholder_: [[0, 0], [0, 0]],
        self.time_shift_offset_placeholder_: [0, 0],
        self.background_data_placeholder_: np.zeros([desired_samples, 1]),
        self.background_volume_placeholder_: 0,
        self.foreground_volume_placeholder_: 1,
    }
    # Run the graph to produce the output audio.
    data_tensor = sess.run([self.output_], feed_dict=input_dict)
    return data_tensor

  def get_unprocessed_data(self, how_many, model_settings, mode):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says labels are one-hot, but they are word strings looked up in
    # the word list.
    # Loads raw (decoded, fixed length) audio for a partition without distortions or features.
    # how_many of -1 means all entries in order; otherwise that many randomly chosen entries (with
    # replacement). Silence entries yield zeros. Uses a throwaway graph and session.
    # Returns a pair: 2D float array (samples by desired_samples) and a list of label strings.
    """Retrieve sample data for the given partition, with no transformations.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      model_settings: Information about the current model being trained.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.

    Returns:
      List of sample data for the samples, and list of labels in one-hot form.
    """
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = how_many
    desired_samples = model_settings['desired_samples']
    words_list = self.words_list
    data = np.zeros((sample_count, desired_samples))
    labels = []
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = tf.audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)
      foreground_volume_placeholder = tf.compat.v1.placeholder(tf.float32, [])
      scaled_foreground = tf.multiply(wav_decoder.audio,
                                      foreground_volume_placeholder)
      for i in range(sample_count):
        if how_many == -1:
          sample_index = i
        else:
          sample_index = np.random.randint(len(candidates))
        sample = candidates[sample_index]
        input_dict = {wav_filename_placeholder: sample['file']}
        if sample['label'] == SILENCE_LABEL:
          input_dict[foreground_volume_placeholder] = 0
        else:
          input_dict[foreground_volume_placeholder] = 1
        data[i, :] = sess.run(scaled_foreground, feed_dict=input_dict).flatten()
        label_index = self.word_to_index[sample['label']]
        labels.append(words_list[label_index])
    return data, labels
