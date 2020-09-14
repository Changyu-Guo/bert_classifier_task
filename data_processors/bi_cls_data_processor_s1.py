# -*- coding: utf - 8 -*-

import json
import random
import collections
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from data_processors.commom import extract_relations_from_init_train_table


class Example:
    def __init__(
            self,
            text,
            question,
            is_valid
    ):
        self.text = text
        self.question = question
        self.is_valid = is_valid


class Feature:
    def __init__(
            self,
            unique_id,
            inputs_ids,
            inputs_mask,
            segment_ids,
            is_valid
    ):
        self.unique_id = unique_id
        self.inputs_ids = inputs_ids
        self.inputs_mask = inputs_mask
        self.segment_ids = segment_ids
        self.is_valid = is_valid


class FeatureWriter:
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
        features['unique_ids'] = create_int_feature([feature.unique_id])
        features['inputs_ids'] = create_int_feature(feature.inputs_ids)
        features['inputs_mask'] = create_int_feature(feature.inputs_mask)
        features['segment_ids'] = create_int_feature(feature.segment_ids)
        features['is_valid'] = create_int_feature([feature.is_valid])

        example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(example.SerializeToString())

    def close(self):
        self._writer.close()


def select_random_items(all_items, current_items, select_num):
    selected_items = set()
    while True:
        index = random.randint(0, len(all_items) - 1)

        if all_items[index] not in current_items:
            selected_items.add(all_items[index])
            current_items.add(all_items[index])

        if len(selected_items) == select_num:
            return list(selected_items)


def read_examples_from_init_train(init_train_path):
    with tf.io.gfile.GFile(init_train_path, mode='r') as reader:
        init_train_examples = json.load(reader)
    reader.close()

    _, all_relations, _, _ = extract_relations_from_init_train_table()
    all_subjects = set()

    # 获取所有的 subjects
    for init_train_example in init_train_examples:
        relations = init_train_example['relations']
        for relation in relations:
            all_subjects.add(relation['subject'])

    examples = []

    question_template = '这句话包含了subject的relation信息'

    for init_train_example in init_train_examples:
        text = init_train_example['text']
        relations = init_train_example['relations']

        current_subjects = set([relation['subject'] for relation in relations])
        current_relations = set([relation['relation'] for relation in relations])

        rest_subjects = all_subjects - current_subjects
        rest_subjects = list(rest_subjects)

        random_relations = select_random_items(
            all_items=all_relations,
            current_items=current_relations,
            select_num=len(relations)
        )

        for relation_index, relation in enumerate(relations):
            s = relation['subject']
            r = relation['relation']

            question = question_template.replace('subject', s).replace('relation', r)

            # 每个 subject 和其对应的 relation 都有一个正样本
            example = Example(
                text=text,
                question=question,
                is_valid=1
            )
            examples.append(example)

            # 每个 subject 都和一个随机的未出现过的 relation 构成一个负样本
            random_relation = random_relations[relation_index]
            question = question_template.replace('subject', s).replace('relation', random_relation)
            example = Example(
                text=text,
                question=question,
                is_valid=0
            )
            examples.append(example)

            # 选择一个随机的未出现过的 subject 和 一个随机的 relation 构成一个负样本
            random_subjects = select_random_items(
                all_items=rest_subjects,
                current_items=set(),
                select_num=random.randint(2, 5)
            )
            for random_subject in random_subjects:
                random_index = random.randint(0, len(all_relations) - 1)
                random_relation = all_relations[random_index]
                question = question_template.replace('subject', random_subject)
                question = question.replace('relation', random_relation)
                example = Example(
                    text=text,
                    question=question,
                    is_valid=0
                )
                examples.append(example)
    print(len(examples))
    return examples


def convert_examples_to_features(
        examples, vocab_file_path, max_seq_len, output_fn
):
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file_path)
    tokenizer.enable_padding(length=max_seq_len)
    tokenizer.enable_truncation(max_length=max_seq_len)

    for example_index, example in enumerate(examples):
        text = example.text
        question = example.question
        is_valid = example.is_valid

        tokenizer_outputs = tokenizer.encode(question, text)

        feature = Feature(
            unique_id=example_index,
            inputs_ids=tokenizer_outputs.ids,
            inputs_mask=tokenizer_outputs.attention_mask,
            segment_ids=tokenizer_outputs.type_ids,
            is_valid=is_valid
        )
        output_fn(feature)


def generate_tfrecord_from_json_file(
        input_file_path,
        vocab_file_path,
        output_file_path,
        max_seq_len
):
    examples = read_examples_from_init_train(
        init_train_path=input_file_path
    )
    writer = FeatureWriter(filename=output_file_path)
    convert_examples_to_features(
        examples,
        vocab_file_path,
        max_seq_len,
        writer.process_feature
    )
    meta_data = {
        'data_size': writer.total_features,
        'max_seq_len': max_seq_len
    }

    writer.close()

    return meta_data

