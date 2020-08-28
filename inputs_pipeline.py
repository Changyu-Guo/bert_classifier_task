# -*- coding: utf - 8 -*-

import json
import collections
import tensorflow as tf


tfrecord_save_path = './datasets/init_train.tfrecord'
desc_save_path = './datasets/desc.json'
MAX_SEQ_LEN = 128


def load_desc(path):
    with tf.io.gfile.GFile(path, mode='r') as reader:
        return json.load(reader)


def read_and_batch_from_tfrecord(
        filename, max_seq_len, num_labels,
        shuffle=True, repeat=True, batch_size=None
):
    dataset = tf.data.TFRecordDataset(filename)

    def _parse_example(example):
        name_to_features = {
            'inputs_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64),
            'inputs_mask': tf.io.FixedLenFeature([max_seq_len], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64),
            'label_ids': tf.io.FixedLenFeature([num_labels], tf.int64)
        }
        parsed_example = tf.io.parse_single_example(example, name_to_features)
        inputs_ids = parsed_example['inputs_ids']
        inputs_mask = parsed_example['inputs_mask']
        segment_ids = parsed_example['segment_ids']
        label_ids = parsed_example['label_ids']

        return (inputs_ids, inputs_mask, segment_ids), label_ids

    dataset = dataset.map(
        _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if shuffle:
        dataset = dataset.shuffle(2020)

    if repeat:
        dataset = dataset.repeat()

    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
