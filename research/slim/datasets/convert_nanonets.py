# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Downloads and converts nanonets data to TFRecords of TF-Example protos.

This module downloads the nanonets data, uncompresses it, reads the files
that make up the nanonets data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
from collections import defaultdict

import tensorflow as tf

from datasets import dataset_utils

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    split_dir = os.path.join(dataset_dir, 'split')
    if not os.path.exists(split_dir):
        raise Exception

    classes = []
    with open(os.path.join(dataset_dir, 'classes.txt'), 'r') as f:
        classes = [klass.strip() for klass in f.readlines()]

    for klass in classes:
        if not os.path.exists(os.path.join(split_dir, klass + '.txt')):
            raise Exception

        if not os.path.exists(os.path.join(split_dir, klass + '_test.txt')):
            raise Exception

    train_filenames = defaultdict(list)
    test_filenames = defaultdict(list)
    for klass in classes:
        with open(os.path.join(split_dir, klass + '.txt'), 'r') as f:
            filenames = f.read().splitlines()
            for filename in filenames:
                if os.path.exists(filename):
                    train_filenames[klass].append(filename)

        with open(os.path.join(split_dir, klass + '_test.txt'), 'r') as f:
            filenames = f.read().splitlines()
            for filename in filenames:
                if os.path.exists(filename):
                    test_filenames[klass].append(filename)

    return train_filenames, test_filenames, classes


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'nanonets_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, 'tfrecord', output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A zip of filename and class
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i+1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        filename, class_name = filenames[i]
                        print(filename, class_name)
                        image_data = tf.gfile.GFile(os.path.join(
                            dataset_dir, filename), 'rb').read()
                        height, width = image_reader.read_image_dims(
                            sess, image_data)

                        class_id = class_names_to_ids[class_name]

                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def run(dataset_dir):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(os.path.join(dataset_dir, 'tfrecord')):
        tf.gfile.MakeDirs(os.path.join(dataset_dir, 'tfrecord'))

    train_filedicts, validation_filedicts, class_names = _get_filenames_and_classes(
        dataset_dir)
    print(class_names)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    train_filenames = []
    validation_filenames = []

    # Divide into train and test:
    for klass in class_names:
        train_classes = [klass]*len(train_filedicts[klass])
        train_filenames.extend(
            list(zip(train_filedicts[klass], train_classes)))

        validation_classes = [klass]*len(validation_filedicts[klass])
        validation_filenames.extend(
            list(zip(validation_filedicts[klass], validation_classes)))

    random.seed(_RANDOM_SEED)
    random.shuffle(train_filenames)
    random.shuffle(validation_filenames)

    # First, convert the training and validation sets.
    _convert_dataset('train', train_filenames, class_names_to_ids,
                     dataset_dir)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    print('\nFinished converting the Nanonets dataset!')
