# -*- coding: utf - 8 -*-

import collections
import tensorflow as tf


tfrecord_save_path = './datasets/init_train.tfrecord'
MAX_SEQ_LEN = 128


class Example:
    def __init__(self, text, relations):
        self.text = text
        self.relations = relations


class Feature:
    def __init__(self, inputs_ids, inputs_mask, segment_ids, label_ids):
        self.inputs_ids = inputs_ids
        self.inputs_mask = inputs_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class FeaturesWriter:
    def __init__(self, filename):
        self.filename = filename
        if tf.io.gfile.exists(filename):
            tf.io.gfile.remove(filename)
        self._writer = tf.io.TFRecordWriter(filename)

    def process_feature(self, feature):

        def create_int_feature(values):
            feat = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return feat

        features = collections.OrderedDict()
        features['inputs_ids'] = create_int_feature(feature.inputs_ids)
        features['inputs_mask'] = create_int_feature(feature.inputs_mask)
        features['segment_ids'] = create_int_feature(feature.segment_ids)
        features['label_ids'] = create_int_feature(feature.label_ids)

        example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(example.SerializeToString())

    def close(self):
        self._writer.close()


def read_and_batch_from_tfrecord(
        filename, max_seq_len, labels_len,
        shuffle=True, repeat=True, batch_size=32
):
    dataset = tf.data.TFRecordDataset(filename)

    def _parse_example(example):
        name_to_features = {
            'inputs_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64),
            'inputs_mask': tf.io.FixedLenFeature([max_seq_len], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64),
            'label_ids': tf.io.FixedLenFeature([labels_len], tf.int64)
        }
        parsed_example = tf.io.parse_single_example(example, name_to_features)
        inputs_ids = parsed_example['inputs_ids']
        inputs_mask = parsed_example['inputs_mask']
        segment_ids = parsed_example['segment_ids']
        label_ids = parsed_example['label_ids']

        return {
            'inputs_ids': inputs_ids,
            'token_type_ids': segment_ids,
            'attention_mask': inputs_mask,
            'label_ids': label_ids
        }

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


if __name__ == '__main__':
    dataset = read_and_batch_from_tfrecord(
        tfrecord_save_path, MAX_SEQ_LEN,
        labels_len=53,
        batch_size=2
    )
    for data in dataset:
        print(data)
        break
