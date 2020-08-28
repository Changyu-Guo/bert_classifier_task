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
from inputs_pipeline import *

init_train_table_txt_path = 'D:\\projects\\2020_08_27_bert_classifier_task\\datasets\\init-train-table.txt'
init_train_txt_path = 'D:\\projects\\2020_08_27_bert_classifier_task\\datasets\\init-train.txt'
tfrecord_save_path = './datasets/init_train.tfrecord'
vocab_filepath = 'D:\\projects\\2020_08_27_bert_classifier_task\\tokenizations\\vocab.txt'

MAX_SEQ_LEN = 128


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


def convert_examples_to_features(examples, labels, save_path, max_seq_len):
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_filepath)
    tokenizer.enable_padding(
        direction='right',
        length=max_seq_len
    )
    tokenizer.enable_truncation(max_seq_len)
    label_to_id_map = get_label_to_id_map(labels)
    labels_len = len(labels)
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
            label_ids=ids_to_vector(text_labels_ids, labels_len)
        )
        tfrecord_writer.process_feature(feature)


def generate_tfrecord():
    _, relations, _, _ = extract_relations_from_init_train_table()
    examples = extract_examples_from_init_train()
    convert_examples_to_features(
        examples,
        relations,
        tfrecord_save_path,
        MAX_SEQ_LEN
    )


if __name__ == '__main__':
    generate_tfrecord()
