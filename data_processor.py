# -*- coding: utf - 8 -*-
"""
    Notice: Bert Version - chinese-bert-base
"""

import os
import json
import time
import collections
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from custom_metrics import compute_prf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from utils import *

init_train_table_txt_path = './datasets/init-train-table.txt'
init_train_txt_path = './datasets/init-train.txt'
tfrecord_save_path = './datasets/init_train.tfrecord'
desc_json_save_path = './datasets/desc.json'
vocab_filepath = './vocab.txt'

MAX_SEQ_LEN = 200


class Example:
    def __init__(self, text, sro_list):
        self.text = text
        self.sro_list = sro_list


class Feature:
    def __init__(
            self, inputs_ids, inputs_mask, segment_ids,
            label_vector, label_ids, start_positions, end_positions
    ):
        self.inputs_ids = inputs_ids
        self.inputs_mask = inputs_mask
        self.segment_ids = segment_ids
        self.label_vector = label_vector
        self.label_ids = label_ids
        self.start_positions = start_positions
        self.end_positions = end_positions


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


def inference(model, ret_path):
    _, relations, _, _ = extract_relations_from_init_train_table()
    id_to_relation_map = relations

    tokenizer = BertWordPieceTokenizer('./vocab.txt')

    tokenizer.enable_padding(length=200)
    tokenizer.enable_truncation(max_length=200)

    checkpoint = tf.train.Checkpoint(model=model)

    checkpoint.restore(tf.train.latest_checkpoint('./saved_models'))

    examples = extract_examples_from_init_train()

    writer = tf.io.gfile.GFile(ret_path, mode='w')

    for index, example in enumerate(examples):
        text = example.text
        relations = example.relations

        tokenizer_outputs = tokenizer.encode(text)
        inputs_ids = tokenizer_outputs.ids
        inputs_mask = tokenizer_outputs.attention_mask
        segment_ids = tokenizer_outputs.type_ids

        inputs_ids = tf.constant(inputs_ids, dtype=tf.int64)
        inputs_mask = tf.constant(inputs_mask, dtype=tf.int64)
        segment_ids = tf.constant(segment_ids, dtype=tf.int64)

        inputs_ids = tf.reshape(inputs_ids, (1, -1))
        inputs_mask = tf.reshape(inputs_mask, (1, -1))
        segment_ids = tf.reshape(segment_ids, (1, -1))

        pred = model.predict(((inputs_ids, inputs_mask, segment_ids),))

        pred = tf.where(pred > 0.5).numpy()

        pred_relations = []
        for p in pred:
            pred_relations.append(id_to_relation_map[p[1]])

        j = {
            "text": text,
            "origin_relations": relations,
            "pred_relations": pred_relations
        }

        writer.write(json.dumps(j, ensure_ascii=False) + '\n')

        if (index + 1) % 500 == 0:
            print(index + 1)


def inference_tfrecord(model, ret_path, dataset):
    _, relations, _, _ = extract_relations_from_init_train_table()
    id_to_relation_map = relations

    tokenizer = BertWordPieceTokenizer(vocab_file='./vocab.txt')
    writer = tf.io.gfile.GFile(ret_path, mode='w')

    index = 0
    y_true, y_pred = [], []
    all_sentences = []
    for data in dataset:

        inputs_ids = data[0][0]
        labels_ids = data[1]

        sentences = [
            tokenizer.decode(input_ids).replace(' ', '')
            for input_ids in inputs_ids
        ]
        all_sentences.extend(sentences)

        pred = model.predict(data)
        pred = tf.where(pred > 0.01, 1, 0)

        y_true.append(tf.reshape(label_ids, (-1,)).numpy())
        y_pred.append(tf.reshape(pred, (-1,)).numpy())

        label_ids = tf.where(data[1] == 1)
        labels = [id_to_relation_map[label_id[1]] for label_id in label_ids]
        pred = tf.where(pred == 1)

        pred_relations = []
        for p in pred:
            pred_relations.append(id_to_relation_map[p[1]])

        j = {
            "text": text,
            "origin_relations": labels,
            "pred_relations": pred_relations
        }

        writer.write(json.dumps(j, ensure_ascii=False) + '\n')

        index += 1
        if index % 100 == 0:
            print(index)


def calculate_tfrecord_prf(model, dataset):
    thresholds = np.arange(0.5, 0, -0.01)
    precisions = []
    recalls = []
    f1_scores = []
    for threshold in thresholds:
        y_true, y_pred = [], []
        for data in dataset:
            labels_ids = data[1]
            y_true.extend(labels_ids.numpy())

            pred = model.predict(data)
            pred = tf.where(pred > threshold, 1, 0)
            y_pred.extend(pred.numpy())

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

        print(threshold)

    print(precisions)
    print(recalls)
    print(f1_scores)

    plt.plot(thresholds, precisions, 'r^-', label='Precision')
    plt.plot(thresholds, recalls, 'go-', label='Recall')
    plt.plot(thresholds, f1_scores, 'b+-', label='F1-Score')

    for threshold, precision in zip(thresholds, precisions):
        plt.text(threshold, precision + 0.01, '%.3f' % precision, ha='center', va='bottom', fontsize=6)
    for threshold, recall in zip(thresholds, recalls):
        plt.text(threshold, recall, '%.3f' % recall, ha='center', va='bottom', fontsize=6)
    for threshold, f1_score in zip(thresholds, f1_scores):
        plt.text(threshold, f1_score - 0.01, '%.3f' % f1_score, ha='center', va='bottom', fontsize=6)

    plt.xlabel('thresholds')
    plt.ylabel('precision - recall - f1-score')

    plt.xlim(thresholds[0] - 0.05, thresholds[-1] + 0.05)
    plt.ylim(0.8, 0.95)

    plt.legend()

    plt.show()


def log_inference_tfrecord_time(model, dataset, batch_size):
    _, relations, _, _ = extract_relations_from_init_train_table()
    id_to_relation_map = relations

    dataset = dataset.take(4000)

    total_batch = 0

    start = time.time()
    pred_relations = []
    for data in dataset:
        total_batch += 1

        # (batch_size, num_labels)
        pred = model.predict(data)

        # convert 0 ~ 1 point to 0 and 1
        pred = tf.where(pred > 0.5, 1, 0)

        # get true label
        pred = tf.where(pred == 1)

    end = time.time()
    print(
        'batch_size: {}, total_batch: {}, total_time: {:.4} s, time_per_batch: {:.4} s, time_per_example: {:.4} '
        's'.format(
            batch_size, total_batch, end - start, (end - start) / total_batch,
            (end - start) / (batch_size * total_batch))
    )


if __name__ == '__main__':
    calculate_tfrecord_prf(None, None)
