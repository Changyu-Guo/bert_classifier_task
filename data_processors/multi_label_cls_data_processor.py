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
from utils.data_utils import get_label_to_id_map
from utils.data_utils import ids_to_vector

vocab_filepath = 'vocabs/bert-base-chinese-vocab.txt'

# raw dataset
init_train_table_txt_path = 'datasets/raw_datasets/init-train-table.txt'
init_train_txt_path = 'datasets/raw_datasets/init-train.txt'

# preprocessed
multi_label_cls_train_save_path = 'datasets/preprocessed_datasets/multi_label_cls_train.json'
multi_label_cls_valid_save_path = 'datasets/preprocessed_datasets/multi_label_cls_valid.json'


# tfrecord
multi_label_cls_train_tfrecord_save_path = 'datasets/tfrecord_datasets/multi_label_cls_train.tfrecord'
multi_label_cls_valid_tfrecord_save_path = 'datasets/tfrecord_datasets/multi_label_cls_valid.tfrecord'

# tfrecord meta
multi_label_cls_train_meta_save_path = 'datasets/tfrecord_datasets/multi_label_cls_train_meta.json'
multi_label_cls_valid_meta_save_path = 'datasets/tfrecord_datasets/multi_label_cls_valid_meta.json'

MAX_SEQ_LEN = 120


class InitTrainExample:
    def __init__(self, text, relations):
        self.text = text
        self.relations = relations


class InitTrainFeature:
    def __init__(
            self,
            unique_id,
            example_index,
            inputs_ids,
            inputs_mask,
            segment_ids,
            label_indices,
            origin_text,
            origin_relations  # 用于推断任务
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.inputs_ids = inputs_ids
        self.inputs_mask = inputs_mask
        self.segment_ids = segment_ids
        self.label_indices = label_indices
        self.origin_text = origin_text
        self.origin_relations = origin_relations


class FeaturesWriter:
    def __init__(self, filename):
        self.filename = filename
        if tf.io.gfile.exists(filename):
            tf.io.gfile.remove(filename)
        self._writer = tf.io.TFRecordWriter(filename)
        self.total_features = 0

    def process_feature(self, feature):
        self.total_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values))
            )
            return feature

        features = collections.OrderedDict()
        # 写入 unique_ids，用于推断任务中
        features['unique_ids'] = create_int_feature([feature.unique_id])
        features['inputs_ids'] = create_int_feature(feature.inputs_ids)
        features['inputs_mask'] = create_int_feature(feature.inputs_mask)
        features['segment_ids'] = create_int_feature(feature.segment_ids)
        features['label_indices'] = create_int_feature(feature.label_indices)

        example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(example.SerializeToString())

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
    """
        用于获取所有的 relation
    """
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

    assert len(relations_set) == len(relations) == len(init_train_table) == len(combined_str_set)

    return subjects, relations, objects, combined_str


def extract_examples_from_init_train():
    """
        Extract Init Train Example
    """
    init_train = load_init_train_txt()
    init_train_examples = []
    for item in init_train:
        item = json.loads(item)
        text = item['text'].strip()
        sro_list = item['sro_list']
        relations = [
            {
                'relation': sro['relation'].strip(),
                'subject': sro['subject'].strip(),
                'object': sro['object'].strip()
            } for sro in sro_list
        ]
        init_train_example = InitTrainExample(text, relations)
        init_train_examples.append(init_train_example)
    return init_train_examples


def split_init_train_data(
        split_valid_ratio=0.05
):
    """
        从 Example 的层面切分数据集
        切分后的数据集保存的是 Example
    """
    split_valid_index = 1 / split_valid_ratio
    init_train_examples = extract_examples_from_init_train()
    multi_label_cls_train = []
    multi_label_cls_valid = []
    for index, example in enumerate(init_train_examples):
        item = {
            'text': example.text,
            'relations': example.relations
        }
        if (index + 1) % split_valid_index == 0:
            multi_label_cls_valid.append(item)
        else:
            multi_label_cls_train.append(item)

    with tf.io.gfile.GFile(multi_label_cls_train_save_path, mode='w') as writer:
        writer.write(json.dumps(multi_label_cls_train, ensure_ascii=False, indent=2))
    writer.close()
    with tf.io.gfile.GFile(multi_label_cls_valid_save_path, mode='w') as writer:
        writer.write(json.dumps(multi_label_cls_valid, ensure_ascii=False, indent=2))
    writer.close()


def read_init_train_examples(filepath):
    """
        读取序列化的 Example
    """
    with tf.io.gfile.GFile(filepath, mode='r') as reader:
        input_data = json.load(reader)

    examples = []
    for item in input_data:
        text = item['text']
        relations = item['relations']
        example = InitTrainExample(text, relations)
        examples.append(example)
    return examples


def convert_examples_to_features(
        examples, vocab_file_path, labels,
        max_seq_len, output_fn
):
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file_path)

    # pad
    tokenizer.enable_padding(length=max_seq_len)

    # trunc
    tokenizer.enable_truncation(max_seq_len)

    label_to_id_map = get_label_to_id_map(labels)

    num_labels = len(labels)

    base_id = 1000000000
    unique_id = base_id

    for example_index, example in enumerate(examples):
        text = example.text
        relations = example.relations

        # label 转为 id
        labels_ids = [label_to_id_map[relation['relation']] for relation in relations]
        # id 转为 indices
        label_indices = ids_to_vector(labels_ids, num_labels)

        tokenizer_outputs = tokenizer.encode(text)

        # 构造 feature
        feature = InitTrainFeature(
            unique_id=unique_id,
            example_index=example_index,
            inputs_ids=tokenizer_outputs.ids,
            inputs_mask=tokenizer_outputs.attention_mask,
            segment_ids=tokenizer_outputs.type_ids,
            label_indices=label_indices,
            origin_text=text,
            origin_relations=relations
        )
        output_fn(feature)
        unique_id += 1


def generate_tfrecord_from_json_file(
        input_file_path,
        vocab_file_path,
        output_file_path,
        max_seq_len=128
):
    _, relations, _, _ = extract_relations_from_init_train_table()
    examples = read_init_train_examples(input_file_path)

    writer = FeaturesWriter(filename=output_file_path)

    convert_examples_to_features(
        examples=examples,
        vocab_file_path=vocab_file_path,
        labels=relations,
        max_seq_len=max_seq_len,
        output_fn=writer.process_feature
    )

    meta_data = {
        'task_type': 'multi_label_classification',
        'data_size': writer.total_features,
        'max_seq_len': max_seq_len,
        'num_labels': len(relations)
    }

    writer.close()

    return meta_data


def postprocess_output(
        all_relations,
        all_features,
        all_results,
        threshold,
        results_save_path
):
    # 一个 feature 拥有一个 unique id
    # 定义一个根据 unique id 找到其结果的映射
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    inference_results = []
    all_origin_indices = []
    all_pred_indices = []
    for feature in all_features:
        unique_id = feature.unique_id
        origin_label_indices = feature.label_indices
        origin_relations = feature.origin_relations
        origin_text = feature.origin_text

        feature_result = unique_id_to_result[unique_id]
        pred_probs = feature_result.probs

        # return a Tensor
        pred_indices = tf.where(pred_probs > threshold, 1, 0)

        all_origin_indices.append(origin_label_indices)
        all_pred_indices.append(pred_indices.numpy().tolist())

        pred_relations = []
        for index, has_relation in enumerate(pred_indices):
            if has_relation == 1:
                pred_relations.append(
                    {
                        'relation': all_relations[index]
                    }
                )

        results_item = {
            'text': origin_text,
            'origin_relations': origin_relations,
            'pred_relations': pred_relations
        }
        inference_results.append(results_item)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_origin_indices, all_pred_indices, average='macro'
    )

    print('precision', precision)
    print('recall', recall)
    print('f1-score', f1)

    with tf.io.gfile.GFile(results_save_path, mode='w') as writer:
        writer.write(json.dumps(inference_results, ensure_ascii=False, indent=2))
    writer.close()


if __name__ == '__main__':
    train_meta_data = generate_tfrecord_from_json_file(
        input_file_path=multi_label_cls_train_save_path,
        vocab_file_path=vocab_filepath,
        output_file_path=multi_label_cls_train_tfrecord_save_path,
        max_seq_len=MAX_SEQ_LEN
    )
    valid_meta_data = generate_tfrecord_from_json_file(
        input_file_path=multi_label_cls_valid_save_path,
        vocab_file_path=vocab_filepath,
        output_file_path=multi_label_cls_valid_tfrecord_save_path,
        max_seq_len=MAX_SEQ_LEN
    )
