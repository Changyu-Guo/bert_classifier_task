# -*- coding: utf - 8 -*-
"""
    Notice: Bert Version - chinese-bert-base
"""

import os
import json
import collections
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from utils import *

init_train_table_txt_path = './datasets/init-train-table.txt'
init_train_txt_path = './datasets/init-train.txt'
tfrecord_save_path = './datasets/init_train.tfrecord'
desc_json_save_path = './datasets/desc.json'
vocab_filepath = './vocab.txt'

MAX_SEQ_LEN = 200


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
        self.total_features = 0

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

        self.total_features += 1

    def save_desc(self, save_filename, **kwargs):
        kwargs['total_features'] = self.total_features
        with tf.io.gfile.GFile(save_filename, mode='w') as writer:
            j = json.dumps(kwargs, indent=2, ensure_ascii=False)
            writer.write(j)
        writer.close()

    def close(self):
        self._writer.close()


def load_init_train_table_txt(filepath=init_train_table_txt_path):
    if not tf.io.gfile.exists(filepath):
        raise ValueError('Init train table txt file {} not found.'.format(filepath))
    with tf.io.gfile.GFile(filepath, mode='r') as reader:
        return reader.readlines()


def load_init_train_txt(filepath=init_train_txt_path):
    if not tf.io.gfile.exists(filepath):
        raise ValueError('Init train txt file {} not found.'.format(filepath))
    with tf.io.gfile.GFile(filepath, mode='r') as reader:
        return reader.readlines()


def extract_relations_from_init_train_table():
    init_train_table = load_init_train_table_txt()
    subjects, relations, objects, combined_str = [], [], [], []
    for item in init_train_table:
        item = item.strip()[1:-2]
        s, r, o = item.split(',')
        identity = ''
        for ls, it in zip((subjects, relations, objects), (s, r, o)):
            it = it.split(':')[1].replace('"', '')
            ls.append(it)
            identity += it
        combined_str.append(identity)

    relations_set = set(relations)
    combined_str_set = set(combined_str)

    assert len(relations_set) == len(relations) == len(init_train_table)
    assert len(combined_str_set) == len(relations)

    return subjects, relations, objects, combined_str


def extract_examples_from_init_train():
    init_train = load_init_train_txt()
    examples = []
    for item in init_train:
        item = json.loads(item)
        text = item['text'].strip()
        sro_list = item['sro_list']
        relations = [sro['relation'] for sro in sro_list]
        example = Example(text, relations)
        examples.append(example)

    return examples


def convert_examples_to_features(
        examples, vocab_file_path, labels,
        max_seq_len, save_path, desc_save_path=None
):
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file_path)

    # pad
    tokenizer.enable_padding(
        direction='right',
        length=max_seq_len
    )

    # trunc
    tokenizer.enable_truncation(max_seq_len)

    label_to_id_map = get_label_to_id_map(labels)

    num_labels = len(labels)
    tfrecord_writer = FeaturesWriter(save_path)
    for example in examples:
        text = example.text
        text_labels = example.relations
        text_labels_ids = [label_to_id_map[label] for label in text_labels]
        tokenizer_outputs = tokenizer.encode(text)
        feature = Feature(
            inputs_ids=tokenizer_outputs.ids,
            inputs_mask=tokenizer_outputs.attention_mask,
            segment_ids=tokenizer_outputs.type_ids,
            label_ids=ids_to_vector(text_labels_ids, num_labels)
        )
        tfrecord_writer.process_feature(feature)

    if desc_save_path is not None:
        kwargs = dict(
            num_labels=num_labels,
            max_seq_len=max_seq_len
        )
        tfrecord_writer.save_desc(desc_save_path, **kwargs)


def generate_tfrecord():
    _, relations, _, _ = extract_relations_from_init_train_table()
    examples = extract_examples_from_init_train()
    convert_examples_to_features(
        examples,
        vocab_filepath,
        relations,
        MAX_SEQ_LEN,
        tfrecord_save_path,
        desc_json_save_path
    )


if __name__ == '__main__':
    generate_tfrecord()
