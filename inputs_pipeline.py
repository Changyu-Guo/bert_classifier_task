# -*- coding: utf - 8 -*-

import tensorflow as tf
from data_processor import Feature, FeaturesWriter


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

    if batch_size:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def split_dataset(dataset, valid_ratio, total_features=None):
    dataset = dataset.shuffle(2020, reshuffle_each_iteration=False)

    if total_features is not None:
        train_dataset_size = int(total_features * (1 - valid_ratio))

        train_dataset = dataset.take(train_dataset_size)
        valid_dataset = dataset.skip(train_dataset_size)

        return train_dataset, valid_dataset

    valid_pick_interval = int(1 / valid_ratio)

    def _is_valid(x, y):
        return x % valid_pick_interval == 0

    def _is_train(x, y):
        return not _is_valid(x, y)

    train_dataset = dataset.enumerate().filter(_is_train).map(lambda x, y: y)
    test_dataset = dataset.enumerate().filter(_is_valid).map(lambda x, y: y)

    return train_dataset, test_dataset


def save_dataset(dataset, path):
    writer = FeaturesWriter(path)
    for data in dataset:
        feature = Feature(
            inputs_ids=data[0][0].numpy(),
            inputs_mask=data[0][1].numpy(),
            segment_ids=data[0][2].numpy(),
            label_ids=data[1].numpy()
        )
        writer.process_feature(feature)

    writer.close()
