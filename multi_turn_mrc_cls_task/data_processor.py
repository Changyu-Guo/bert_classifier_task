# -*- coding: utf - 8 -*-

import json
import gzip
import pickle
import random
import collections
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from common_data_utils import get_squad_json_template
from common_data_utils import get_squad_json_paragraph_template
from common_data_utils import extract_relations_from_init_train_table
from sklearn.metrics import precision_recall_fscore_support

INIT_TRAIN_TRAIN_PATH = '../common-datasets/init-train-train.json'
INIT_TRAIN_VALID_PATH = '../common-datasets/init-train-valid.json'

INIT_TRAIN_TRAIN_SQUAD_SAVE_PATH = 'datasets/raw/train.json'
INIT_TRAIN_VALID_SQUAD_SAVE_PATH = 'datasets/raw/valid.json'


def convert_init_train_to_squad_format(init_train_path, save_path):
    with tf.io.gfile.GFile(init_train_path, mode='r') as reader:
        init_train_examples = json.load(reader)
    reader.close()

    squad_json = get_squad_json_template(title='multi turn mrc cls')
    for init_train_example in init_train_examples:
        text = init_train_example['text']
        sros = init_train_example['sros']
        squad_json_paragraph = get_squad_json_paragraph_template(
            text=text,
            origin_sros=sros,
            pred_sros=[]
        )
        squad_json['data'][0]['paragraphs'].append(squad_json_paragraph)

    with tf.io.gfile.GFile(save_path, mode='w') as writer:
        writer.write(json.dumps(squad_json, ensure_ascii=False, indent=2) + '\n')
    writer.close()


# 每个 relation 对应一个 example
# 每个 relation example 保存自己所属的 example index
class Example:
    def __init__(
            self,
            paragraph_index,
            text,
            question,
            is_valid
    ):
        self.paragraph_index = paragraph_index
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
        features['example_indices'] = create_int_feature([feature.example_index])
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


def read_examples_for_train(filepath):
    """
        从 init train 中读取数据，将其转换为当前任务所需的数据形式
    """
    # 读取 init train 中的数据并转为 json 对象
    with tf.io.gfile.GFile(filepath, mode='r') as reader:
        input_data = json.load(reader)['data']
    reader.close()

    # 53, 53, _, _
    all_subjects, all_relations, _, _ = extract_relations_from_init_train_table(
        '../common-datasets/init-train-table.txt'
    )

    # 根据 relation 获取当前 relation 的 index
    relation_to_index_map = collections.OrderedDict()
    for index, item in enumerate(all_relations):
        relation_to_index_map[item] = index

    # 所有的 subject 类型，理论上应该是 4 个
    subjects_type = set(all_subjects)

    examples = []

    question_template = '这句话包含了subject的relation信息'

    # 对每一条 example 处理
    for index, paragraph in enumerate(input_data[0]['paragraphs']):
        text = paragraph['context']

        sros = paragraph['origin_sros']

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
                paragraph_index=index,
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
                    paragraph_index=index,
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
                        paragraph_index=index,
                        text=text,
                        question=question,
                        is_valid=0
                    )
                    examples.append(example)

    print(len(examples))

    return examples


def read_examples_for_valid(filepath):
    # 读取 init train 中的数据并转为 json 对象
    with tf.io.gfile.GFile(filepath, mode='r') as reader:
        input_data = json.load(reader)['data']
    reader.close()

    # 53, 53, _, _
    all_subjects, all_relations, _, _ = extract_relations_from_init_train_table(
        '../common-datasets/init-train-table.txt'
    )

    # 根据 relation 获取当前 relation 的 index
    relation_to_index_map = collections.OrderedDict()
    for index, item in enumerate(all_relations):
        relation_to_index_map[item] = index

    examples = []

    question_template = '这句话包含了subject的relation信息'

    # 对每一条 example 处理
    for index, paragraph in enumerate(input_data[0]['paragraphs']):

        text = paragraph['context']

        sros = paragraph['origin_sros']

        current_relations = list(set([sro['relation'] for sro in sros]))

        for relation, subject in zip(all_relations, all_subjects):
            question = question_template.replace('subject', subject).replace('relation', relation)
            # 当前样本拥有的 relation，应预测为 1
            if relation in current_relations:
                example = Example(
                    paragraph_index=index,
                    text=text,
                    question=question,
                    is_valid=1
                )
            else:
                # 当前样本没有的 relation，应预测为 0
                example = Example(
                    paragraph_index=index,
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

        if (example_index + 1) % 1000 == 0:
            print(example_index + 1)


def generate_tfrecord_from_json_file(input_file_path, vocab_file_path,
                                     output_save_path, meta_save_path,
                                     features_save_path, max_seq_len,
                                     is_train):
    if is_train:
        examples = read_examples_for_train(
            filepath=input_file_path
        )
    else:
        examples = read_examples_for_valid(
            filepath=input_file_path
        )

    writer = FeatureWriter(filename=output_save_path)

    features = []

    def _append_feature(feature):
        features.append(feature)
        writer.process_feature(feature)

    convert_examples_to_features(
        examples=examples,
        vocab_file_path=vocab_file_path,
        max_seq_len=max_seq_len,
        output_fn=_append_feature
    )
    meta_data = {
        'data_size': writer.total_features,
        'max_seq_len': max_seq_len
    }
    writer.close()

    # save meta info
    with tf.io.gfile.GFile(meta_save_path, mode='w') as writer:
        writer.write(json.dumps(meta_data, ensure_ascii=False, indent=2) + '\n')
    writer.close()

    # save features
    with gzip.open(features_save_path, 'wb') as writer:
        pickle.dump(features, writer, protocol=pickle.HIGHEST_PROTOCOL)
    writer.close()


def postprocess_results(
        raw_data_path,
        features_path,
        results_path,
        save_path
):
    _, relations, _, _ = extract_relations_from_init_train_table('../common-datasets/init-train-table.txt')

    with tf.io.gfile.GFile(raw_data_path, mode='r') as reader:
        raw_data = json.load(reader)
    reader.close()
    paragraphs = raw_data['data'][0]['paragraphs']

    with gzip.open(features_path, mode='rb') as reader:
        features = pickle.load(reader)
    reader.close()

    with tf.io.gfile.GFile(results_path, mode='r') as reader:
        results = json.load(reader)
    reader.close()

    assert len(results) % 53 == 0
    assert len(results) / 53 == len(paragraphs)

    for index, paragraph in enumerate(paragraphs):

        result_pos = index * 53
        for i in range(53):
            result_index = result_pos + i
            result = results[result_index]
            prob = result['prob']
            if prob >= 0.5:
                paragraphs[index]['pred_sros'].append({
                    'relation': relations[i]
                })

    raw_data['data'][0]['paragraphs'] = paragraphs
    with tf.io.gfile.GFile(save_path, mode='w') as writer:
        writer.write(json.dumps(raw_data, ensure_ascii=False, indent=2))
    writer.close()


def compute_prf(features_path, results_path):
    with gzip.open(features_path, mode='rb') as reader:
        features = pickle.load(reader)
    reader.close()

    with tf.io.gfile.GFile(results_path, mode='r') as reader:
        results = json.load(reader)
    reader.close()

    origin_is_valid = []
    pred_is_valid = []
    for feature, result in zip(features, results):
        origin_is_valid.append(feature.is_valid)
        if result['prob'] >= 0.5:
            pred_is_valid.append(1)
        else:
            pred_is_valid.append(0)

    precision, recall, f_score, _ = precision_recall_fscore_support(
        y_true=origin_is_valid,
        y_pred=pred_is_valid,
        average='binary'
    )

    print('precision: ', precision)
    print('recall: ', recall)
    print('f1-score: ', f_score)


if __name__ == '__main__':
    # generate train tfrecord for train
    # generate_tfrecord_from_json_file(
    #     input_file_path='datasets/raw/train.json',
    #     vocab_file_path='../vocabs/bert-base-chinese-vocab.txt',
    #     output_save_path='datasets/tfrecords/for_train/train.tfrecord',
    #     meta_save_path='datasets/tfrecords/for_train/train_meta.json',
    #     features_save_path='datasets/features/for_train/train_features.pkl',
    #     max_seq_len=165,
    #     is_train=True  # 含有随机的样本, 用于训练
    # )

    # generate train tfrecord for infer
    # generate_tfrecord_from_json_file(
    #     input_file_path='datasets/raw/train.json',
    #     vocab_file_path='../vocabs/bert-base-chinese-vocab.txt',
    #     output_save_path='datasets/tfrecords/origin/train.tfrecord',
    #     meta_save_path='datasets/tfrecords/origin/train_meta.json',
    #     features_save_path='datasets/features/origin/train_features.pkl',
    #     max_seq_len=165,
    #     is_train=False  # infer 的过程中不需要使用随机的样本, 就对固定的 53 个问题进行推断
    # )

    # generate valid tfrecord for train and infer
    # (valid data 无论是 train 还是 infer 都是一样的)
    # generate_tfrecord_from_json_file(
    #     input_file_path='datasets/raw/valid.json',
    #     vocab_file_path='../vocabs/bert-base-chinese-vocab.txt',
    #     output_save_path='datasets/tfrecords/for_train/valid.tfrecord',
    #     meta_save_path='datasets/tfrecords/for_train/valid_meta.json',
    #     features_save_path='datasets/features/for_train/valid_features.pkl',
    #     max_seq_len=165,
    #     is_train=False  # 这里一定要为 False, 因为 valid data 不需要随机的样本
    # )

    # generate_tfrecord_from_json_file(
    #     input_file_path='datasets/raw/valid.json',
    #     vocab_file_path='../vocabs/bert-base-chinese-vocab.txt',
    #     output_save_path='datasets/tfrecords/origin/valid.tfrecord',
    #     meta_save_path='datasets/tfrecords/origin/valid_meta.json',
    #     features_save_path='datasets/features/origin/valid_features.pkl',
    #     max_seq_len=165,
    #     is_train=False  # 这里一定要为 False, 因为 valid data 不需要随机的样本
    # )

    # 处理训练数据的推断结果
    # postprocess_results(
    #     raw_data_path='datasets/raw/train.json',
    #     features_path='datasets/features/origin/train_features.pkl',
    #     results_path='infer_results/origin/raw/train_results.json',
    #     save_path='infer_results/origin/postprocessed/train_results.json'
    # )

    # 处理验证集的推断结果
    # postprocess_results(
    #     raw_data_path='datasets/raw/valid.json',
    #     features_path='datasets/features/origin/valid_features.pkl',
    #     results_path='infer_results/origin/raw/valid_results.json',
    #     save_path='infer_results/origin/postprocessed/valid_results.json'
    # )

    # 训练集 PRF
    compute_prf(
        features_path='datasets/features/for_infer/train_features.pkl',
        results_path='infer_results/origin/raw/train_results.json'
    )

    # 测试机 PRF
    compute_prf(
        features_path='datasets/features/for_infer/valid_features.pkl',
        results_path='infer_results/origin/raw/valid_results.json'
    )
