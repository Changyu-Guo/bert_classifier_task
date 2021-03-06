# -*- coding: utf - 8 -*-

import json
import random
import collections
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from data_processors.commom import read_init_train_valid_examples
from data_processors.commom import extract_relations_from_init_train_table
from sklearn.metrics import precision_recall_fscore_support


# 每个 relation 对应一个 example
# 每个 relation example 保存自己所属的 example index
class Example:
    def __init__(
            self,
            init_train_example_index,
            text,
            question,
            is_valid
    ):
        self.init_train_example_index = init_train_example_index
        self.text = text
        self.question = question
        self.is_valid = is_valid


# relation 的 feature
# 一个 relation example 构造成一个 feature
# 每个 feature 保存其变换之前的 example 的 index
class Feature:
    def __init__(
            self,
            unique_id,
            example_index,
            inputs_ids,
            inputs_mask,
            segment_ids,
            is_valid
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.inputs_ids = inputs_ids
        self.inputs_mask = inputs_mask
        self.segment_ids = segment_ids
        self.is_valid = is_valid


class FeatureWriter:
    def __init__(self, filename):
        self.filename = filename

        # 移除之前的 tfrecord file
        # TODO 考察是否真的需要这一步骤
        if tf.io.gfile.exists(filename):
            tf.io.gfile.remove(filename)

        # 打开文件
        options = tf.io.TFRecordOptions(compression_type='GZIP')
        self._writer = tf.io.TFRecordWriter(filename, options)
        self.total_features = 0

    def process_feature(self, feature):
        self.total_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values))
            )
            return feature

        # 在任何一个 FeatureWriter 中
        # 理论上应该只需要修改这里的代码
        # 以确保 TFRecord 中保存的内容以及保存格式
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


def select_random_items(all_items, select_num):
    selected_items = set()
    while True:
        index = random.randint(0, len(all_items) - 1)

        if all_items[index] not in selected_items:
            selected_items.add(all_items[index])

        if len(selected_items) == select_num:
            return list(selected_items)


def read_train_examples_from_init_train(init_train_path):
    """
        从 init train 中读取数据，将其转换为当前任务所需的数据形式
    """
    # 读取 init train 中的数据并转为 json 对象
    with tf.io.gfile.GFile(init_train_path, mode='r') as reader:
        init_train_examples = json.load(reader)
    reader.close()

    # 53, 53, _, _
    all_subjects, all_relations, _, _ = extract_relations_from_init_train_table()

    # 根据 relation 获取当前 relation 的 index
    relation_to_index_map = collections.OrderedDict()
    for index, item in enumerate(all_relations):
        relation_to_index_map[item] = index

    # 所有的 subject 类型，理论上应该是 4 个
    subjects_type = set(all_subjects)

    examples = []

    question_template = '这句话包含了subject的relation信息'

    # 对每一条 example 处理
    for init_train_example_index, init_train_example in enumerate(init_train_examples):

        text = init_train_example['text']

        sros = init_train_example['sros']

        current_relations = list(set([sro['relation'] for sro in sros]))
        current_relation_indices = [relation_to_index_map[relation] for relation in current_relations]

        # 当前 example 中所有出现的 subject
        current_subjects = set(all_subjects[index] for index in current_relation_indices)

        rest_subjects = subjects_type - current_subjects
        rest_subjects = list(rest_subjects)

        rest_relations = set(all_relations) - set(current_relations)
        rest_relations = list(rest_relations)

        for sro_index, sro in enumerate(sros):
            r = sro['relation']

            r_index = relation_to_index_map[r]
            s_type = all_subjects[r_index]

            # subject 替换为 subject_type
            # relation 替换为当前的 relation
            question = question_template.replace('subject', s_type).replace('relation', r)

            # 每个 subject 和其对应的 relation 都有一个正样本
            example = Example(
                init_train_example_index=init_train_example_index,
                text=text,
                question=question,
                is_valid=1
            )
            examples.append(example)

            # 每个已出现过的 subject 都和若干个随机的未出现过的 relation 构成一个负样本
            random_relations = select_random_items(
                all_items=rest_relations,
                select_num=random.randint(2, 5)
            )
            for random_relation in random_relations:
                question = question_template.replace('subject', s_type).replace('relation', random_relation)
                example = Example(
                    init_train_example_index=init_train_example_index,
                    text=text,
                    question=question,
                    is_valid=0
                )
                examples.append(example)

        if len(rest_subjects) != 0:
            # 每一个未出现过的 subject 和 若干个随机的 relation 构成一个负样本
            random_subjects = rest_subjects
            for random_subject in random_subjects:

                # 若干个未出现过的 relation
                random_relations = select_random_items(
                    all_items=rest_relations,
                    select_num=random.randint(2, 5)
                )
                for random_relation in random_relations:
                    question = question_template.replace('subject', random_subject)
                    question = question.replace('relation', random_relation)
                    example = Example(
                        init_train_example_index=init_train_example_index,
                        text=text,
                        question=question,
                        is_valid=0
                    )
                    examples.append(example)

    print(len(examples))

    return examples


def read_valid_examples_from_init_train(init_train_path):
    # 读取 init train 中的数据并转为 json 对象
    with tf.io.gfile.GFile(init_train_path, mode='r') as reader:
        init_train_examples = json.load(reader)
    reader.close()

    # 53, 53, _, _
    all_subjects, all_relations, all_objects, _ = extract_relations_from_init_train_table()

    # 根据 relation 获取当前 relation 的 index
    relation_to_index_map = collections.OrderedDict()
    for index, item in enumerate(all_relations):
        relation_to_index_map[item] = index

    examples = []

    question_template = '这句话包含了subject的relation信息'

    # 对每一条 example 处理
    for init_train_example_index, init_train_example in enumerate(init_train_examples):

        text = init_train_example['text']

        sros = init_train_example['sros']

        current_relations = list(set([sro['relation'] for sro in sros]))

        for relation, subject in zip(all_relations, all_subjects):
            question = question_template.replace('subject', subject).replace('relation', relation)
            # 当前样本拥有的 relation，应预测为 1
            if relation in current_relations:
                example = Example(
                    init_train_example_index=init_train_example_index,
                    text=text,
                    question=question,
                    is_valid=1
                )
            else:
                # 当前样本没有的 relation，应预测为 0
                example = Example(
                    init_train_example_index=init_train_example_index,
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
    """
        此函数的主要作用是对 text tokenize 并转为 ids

        TODO: 学习如何单纯使用 huggingface/tokenizers 来编写此函数
    """
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file_path)
    tokenizer.enable_padding(length=max_seq_len)
    tokenizer.enable_truncation(max_length=max_seq_len)

    base_id = 1000000000
    unique_id = base_id
    for example_index, example in enumerate(examples):
        text = example.text
        question = example.question
        is_valid = example.is_valid

        tokenizer_outputs = tokenizer.encode(question, text)

        feature = Feature(
            unique_id=unique_id,
            example_index=example_index,
            inputs_ids=tokenizer_outputs.ids,
            inputs_mask=tokenizer_outputs.attention_mask,
            segment_ids=tokenizer_outputs.type_ids,
            is_valid=is_valid
        )

        unique_id += 1

        output_fn(feature)


def generate_tfrecord_from_json_file(
        input_file_path,
        vocab_file_path,
        output_file_path,
        max_seq_len,
        is_train,
        return_examples_and_features
):
    if is_train:
        examples = read_train_examples_from_init_train(
            init_train_path=input_file_path
        )
    else:
        examples = read_valid_examples_from_init_train(
            init_train_path=input_file_path
        )

    writer = FeatureWriter(filename=output_file_path)

    features = []

    def _append_feature(feature):
        features.append(feature)
        writer.process_feature(feature)

    if return_examples_and_features:
        output_fn = _append_feature
    else:
        output_fn = writer.process_feature

    convert_examples_to_features(
        examples=examples,
        vocab_file_path=vocab_file_path,
        max_seq_len=max_seq_len,
        output_fn=output_fn
    )
    meta_data = {
        'data_size': writer.total_features,
        'max_seq_len': max_seq_len
    }
    writer.close()

    if return_examples_and_features:
        return meta_data, examples, features
    else:
        return meta_data


def postprocess_valid_output(
        batched_origin_is_valid,
        batched_pred_is_valid,
        results_save_path
):
    precision, recall, f1, _ = precision_recall_fscore_support(
        batched_origin_is_valid, batched_pred_is_valid, average='micro'
    )
    print(precision)
    print(recall)
    print(f1)

    # origin valid data
    init_train_examples = read_init_train_valid_examples('common-datasets/raw_datasets/init-train-init-train-valid-squad-format.json')
    _, relations, _, _ = extract_relations_from_init_train_table()

    # 每一条 example 都对应一个原始的 init train example
    for example_index, example in enumerate(init_train_examples):

        # 当前 example / feature 对应的原始 valid data index
        init_train_example_index = example_index

        # 构建 pred_sros 键
        if not init_train_examples[init_train_example_index].get('pred_sros'):
            init_train_examples[init_train_example_index]['pred_sros'] = []

        # 获取当前 init train example 对应的一组推断结果
        current_pred_indices = batched_pred_is_valid[init_train_example_index]

        # 将 53 个预测结果对应到 relation
        # 然后添加到对应的 init train example 中
        for i in range(len(current_pred_indices)):
            if current_pred_indices[i] == 1:
                init_train_examples[init_train_example_index]['pred_sros'].append(
                    {
                        'relation': relations[i]
                    }
                )

    with tf.io.gfile.GFile(results_save_path, mode='w') as writer:
        writer.write(json.dumps(init_train_examples, ensure_ascii=False, indent=2))
    writer.close()


def postprocess_train_output(
        batched_origin_is_valid,
        batched_pred_is_valid,
        results_save_path
):
    precision, recall, f1, _ = precision_recall_fscore_support(
        batched_origin_is_valid, batched_pred_is_valid, average='micro'
    )
    print(precision)
    print(recall)
    print(f1)

    # origin valid data
    init_train_examples = read_init_train_valid_examples('common-datasets/raw_datasets/init-train-init-train-train-squad-format.json')
    _, relations, _, _ = extract_relations_from_init_train_table()

    # 每一条 example 都对应一个原始的 init train example
    for example_index, example in enumerate(init_train_examples):

        # 当前 example / feature 对应的原始 valid data index
        init_train_example_index = example_index

        # 构建 pred_sros 键
        if not init_train_examples[init_train_example_index].get('pred_sros'):
            init_train_examples[init_train_example_index]['pred_sros'] = []

        # 获取当前 init train example 对应的一组推断结果
        current_pred_indices = batched_pred_is_valid[init_train_example_index]

        # 将 53 个预测结果对应到 relation
        # 然后添加到对应的 init train example 中
        for i in range(len(current_pred_indices)):
            if current_pred_indices[i] == 1:
                init_train_examples[init_train_example_index]['pred_sros'].append(
                    {
                        'relation': relations[i]
                    }
                )

    with tf.io.gfile.GFile(results_save_path, mode='w') as writer:
        writer.write(json.dumps(init_train_examples, ensure_ascii=False, indent=2))
    writer.close()
