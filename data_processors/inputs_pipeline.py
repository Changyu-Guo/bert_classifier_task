# -*- coding: utf - 8 -*-

import json
import tensorflow as tf


def read_and_batch_from_multi_label_cls_tfrecord(
        filename, max_seq_len, num_labels,
        shuffle=True, repeat=False, batch_size=None
):
    name_to_features = {
        'unique_ids': tf.io.FixedLenFeature([], tf.int64),
        'inputs_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'inputs_mask': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'label_indices': tf.io.FixedLenFeature([num_labels], tf.int64)
    }

    dataset = tf.data.TFRecordDataset(filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2020)

    if repeat:
        dataset = dataset.repeat()

    def _parse_example(example):
        parsed_example = tf.io.parse_single_example(example, name_to_features)

        return parsed_example

    dataset = dataset.map(
        _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if batch_size:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def read_and_batch_from_squad_tfrecord(
        filename,
        max_seq_len,
        is_training,
        repeat=False,
        batch_size=None
):
    name_to_features = {
        'unique_ids': tf.io.FixedLenFeature([], tf.int64),
        'input_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'input_mask': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64)
    }
    if is_training:
        name_to_features['start_positions'] = tf.io.FixedLenFeature([], tf.int64)
        name_to_features['end_positions'] = tf.io.FixedLenFeature([], tf.int64)

    dataset = tf.data.TFRecordDataset(filename)
    if is_training:
        dataset = dataset.shuffle(buffer_size=100)

    if repeat:
        dataset = dataset.repeat()

    def _parse_example(example):
        parsed_example = tf.io.parse_single_example(example, name_to_features)

        return parsed_example

    dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if batch_size:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def read_and_batch_from_bi_cls_record(
        filename,
        max_seq_len,
        repeat=False,
        shuffle=False,
        batch_size=None
):
    name_to_features = {
        'inputs_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'inputs_mask': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'is_valid': tf.io.FixedLenFeature([], tf.int64)
    }

    dataset = tf.data.TFRecordDataset(filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2020)

    if repeat:
        dataset = dataset.repeat()

    def _parse_example(example):
        parsed_example = tf.io.parse_single_example(example, name_to_features)

        return parsed_example

    dataset = dataset.map(
        _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if batch_size:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def map_data_to_multi_label_cls_train_task(data):
    x = {
        'inputs_ids': data['inputs_ids'],
        'inputs_mask': data['inputs_mask'],
        'segment_ids': data['segment_ids']
    },
    y = {
        'probs': data['label_indices']
    }
    return x, y


def map_data_to_multi_label_cls_predict_task(data):
    x = {
        'inputs_ids': data['inputs_ids'],
        'inputs_mask': data['inputs_mask'],
        'segment_ids': data['segment_ids']
    }
    return x


def map_data_to_mrc_train_task(data):
    x = {
        'inputs_ids': data['inputs_ids'],
        'inputs_mask': data['inputs_mask'],
        'segment_ids': data['segment_ids']
    }
    y = {
        'start_logits': data['start_positions'],
        'end_logits': data['end_positions']
    }
    return x, y


def map_data_to_mrc_predict_task(data):
    return {
        'inputs_ids': data['inputs_ids'],
        'inputs_mask': data['inputs_mask'],
        'segment_ids': data['segment_ids']
    }


def map_data_to_bi_cls_train_task(data):
    x = {
        'inputs_ids': data['inputs_ids'],
        'inputs_mask': data['inputs_mask'],
        'segment_ids': data['segment_ids']
    }
    y = {
        'probs': data['is_valid']
    }

    return x, y
