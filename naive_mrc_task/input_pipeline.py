# -*- coding: utf - 8 -*-

import tensorflow as tf


def read_and_batch_from_tfrecord(
        filepath,
        max_seq_len,
        repeat=False,
        shuffle=False,
        batch_size=None
):
    name_to_features = {
        'unique_ids': tf.io.FixedLenFeature([], tf.int64),
    }


def map_data_to_model(data):
    x = {
        'inputs_ids': data['inputs_ids'],
        'inputs_mask': data['inputs_mask'],
        'segment_ids': data['segment_ids']
    }
    y = {
        'start_logits': data['start_logits'],
        'end_logits': data['end_logits']
    }

    return x, y
