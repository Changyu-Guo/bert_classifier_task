# -*- coding: utf - 8 -*-

import copy
import gzip
import json
import pickle
import collections
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from common_data_utils import extract_examples_dict_from_relation_questions


def convert_last_step_results_for_train(results_path, save_path):
    relation_questions_dict = extract_examples_dict_from_relation_questions(
        init_train_table_path='../common-datasets/init-train-table.txt',
        relation_questions_path='../common-datasets/relation_questions.txt'
    )

    # 加载上一步骤的推断结果
    with tf.io.gfile.GFile(results_path, mode='r') as reader:
        results = json.load(reader)
    reader.close()

    paragraphs = results['data'][0]['paragraphs']

    qas_id = 0
    for paragraph_index, paragraph in enumerate(paragraphs):

        paragraphs[paragraph_index]['qas'] = []

        context = paragraph['context']

        origin_sros = paragraph['origin_sros']
        origin_relations = set([sro['relation'] for sro in origin_sros])

        pred_sros = paragraph['pred_sros']

        for index, sro in enumerate(origin_sros):
            s = sro['subject']
            r = sro['relation']
            o = sro['object']

            relation_questions = relation_questions_dict[r]
            question_c = relation_questions.question_c
            question_c = question_c.replace('subject', s).replace('object', o)

            squad_json_qas_item = {
                'question': question_c,
                'context': context,
                'is_valid': 1,
                'qas_id': 'id_' + str(qas_id)
            }
            qas_id += 1
            paragraphs[paragraph_index]['qas'].append(squad_json_qas_item)

        for index, sro in enumerate(pred_sros):
            r = sro['relation']

            if r in origin_relations:
                continue

            s = sro['subject']
            o = sro['object']

            relation_questions = relation_questions_dict[r]
            question_c = relation_questions.question_c
            question_c = question_c.replace('subject', s).replace('object', o)

            squad_json_qas_item = {
                'question': question_c,
                'context': context,
                'is_valid': 0,
                'qas_id': 'id_' + str(qas_id)
            }
            qas_id += 1
            paragraphs[paragraph_index]['qas'].append(squad_json_qas_item)

    results['data'][0]['paragraphs'] = paragraphs
    with tf.io.gfile.GFile(save_path, mode='w') as writer:
        writer.write(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    writer.close()


def convert_last_step_results_for_infer(results_path, save_path):

    relation_questions_dict = extract_examples_dict_from_relation_questions(
        init_train_table_path='../common-datasets/init-train-table.txt',
        relation_questions_path='../common-datasets/relation_questions.txt'
    )

    # 获取上一步骤的推断结果
    with tf.io.gfile.GFile(results_path, mode='r') as reader:
        results = json.load(reader)
    reader.close()

    paragraphs = results['data'][0]['paragraphs']

    qas_id = 0

    for paragraph_index, paragraph in enumerate(paragraphs):

        # 重置之前的 qas
        paragraphs[paragraph_index]['qas'] = []

        context = paragraph['context']

        pred_sros = paragraph['pred_sros']

        # 构建样本
        for index, sro in enumerate(pred_sros):
            s = sro['subject']
            r = sro['relation']
            o = sro['object']

            relation_questions = relation_questions_dict[r]
            question_c = relation_questions.question_c.replace('subject', s).replace('object', o)

            squad_json_qas_item = {
                'question': question_c,
                'context': context,
                'id': 'id_' + str(qas_id),
                'sro_index': index,
            }
            qas_id += 1
            paragraphs[paragraph_index]['qas'].append(squad_json_qas_item)

    results['data'][0]['paragraphs'] = paragraphs

    with tf.io.gfile.GFile(save_path, mode='w') as writer:
        writer.write(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    writer.close()


class BiCLSExample:
    def __init__(self, paragraph_index, sro_index, text, question, is_valid):
        self.paragraph_index = paragraph_index
        self.sro_index = sro_index
        self.text = text
        self.question = question
        self.is_valid = is_valid


class BiCLSFeature:
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
    def __init__(self, filename, is_training):
        self.filename = filename

        options = tf.io.TFRecordOptions(compression_type='GZIP')
        self._writer = tf.io.TFRecordWriter(filename, options=options)

        self.is_training = is_training

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
        features['example_indices'] = create_int_feature([feature.example_index])
        features['inputs_ids'] = create_int_feature(feature.inputs_ids)
        features['inputs_mask'] = create_int_feature(feature.inputs_mask)
        features['segment_ids'] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features['is_valid'] = create_int_feature([feature.is_valid])

        example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(example.SerializeToString())

    def close(self):
        self._writer.close()


def read_bi_cls_examples(filepath, is_training):
    with tf.io.gfile.GFile(filepath, mode='r') as reader:
        data = json.load(reader)
    reader.close()

    examples = []
    paragraphs = data['data'][0]['paragraphs']

    for paragraph_index, paragraph in enumerate(paragraphs):

        for qa in paragraph['qas']:
            text = qa['context']
            question = qa['question']
            is_valid = None
            sro_index = None

            if is_training:
                is_valid = qa['is_valid']
            else:
                sro_index = qa['sro_index']

            example = BiCLSExample(
                paragraph_index=paragraph_index,
                sro_index=sro_index,
                text=text,
                question=question,
                is_valid=is_valid
            )

            examples.append(example)

    return examples


def convert_examples_to_features(
        examples, vocab_file_path, max_seq_len, output_fn
):
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file_path)
    tokenizer.enable_padding(length=max_seq_len)
    tokenizer.enable_truncation(max_seq_len)

    base_id = 1000000000
    unique_id = base_id
    for example_index, example in enumerate(examples):
        text = example.text
        question = example.question
        is_valid = example.is_valid

        tokenizer_outputs = tokenizer.encode(question, text)

        feature = BiCLSFeature(
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
        tfrecord_save_path,
        meta_save_path,
        features_save_path,
        max_seq_len=128,
        is_train=True
):
    examples = read_bi_cls_examples(input_file_path, is_training=is_train)
    writer = FeatureWriter(filename=tfrecord_save_path, is_training=is_train)

    features = []

    def _append_features(feature):
        features.append(feature)
        writer.process_feature(feature)

    convert_examples_to_features(
        examples=examples,
        vocab_file_path=vocab_file_path,
        max_seq_len=max_seq_len,
        output_fn=_append_features
    )
    meta_data = {
        'data_size': writer.total_features,
        'max_seq_len': max_seq_len,
        'is_train': is_train
    }

    writer.close()
    with tf.io.gfile.GFile(meta_save_path, mode='w') as writer:
        writer.write(json.dumps(meta_data, ensure_ascii=False, indent=2) + '\n')
    writer.close()

    with gzip.open(features_save_path, mode='wb') as writer:
        pickle.dump(features, writer, protocol=pickle.HIGHEST_PROTOCOL)
    writer.close()


def postprocess_results(
        raw_data_path,
        features_path,
        results_path,
        save_path
):
    with tf.io.gfile.GFile(raw_data_path, mode='r') as reader:
        raw_data = json.load(reader)
    reader.close()
    paragraphs = raw_data['data'][0]['paragraphs']
    new_paragraphs = copy.deepcopy(paragraphs)

    # flush new paragraphs pred sros
    for i in range(len(new_paragraphs)):
        new_paragraphs[i]['pred_sros'] = []

    examples = read_bi_cls_examples(
        filepath=raw_data_path,
        is_training=False
    )

    with gzip.open(features_path, mode='r') as reader:
        features = pickle.load(reader)
    reader.close()

    with tf.io.gfile.GFile(results_path, mode='r') as reader:
        results = json.load(reader)
    reader.close()

    assert len(examples) == len(features) == len(results)

    example_index_to_features = {}
    for feature in features:
        example_index_to_features[feature.example_index] = feature

    unique_id_to_result = {}
    for result in results:
        unique_id_to_result[result['unique_id']] = result

    for example_index, example in enumerate(examples):

        cur_example_feature = example_index_to_features[example_index]

        result = unique_id_to_result[cur_example_feature.unique_id]

        paragraph_index = example.paragraph_index
        sro_index = example.sro_index

        prob = result['prob']
        if prob >= 0.3:
            new_paragraphs[paragraph_index]['pred_sros'].append(
                paragraphs[paragraph_index]['pred_sros'][sro_index]
            )

    raw_data['data'][0]['paragraphs'] = new_paragraphs

    with tf.io.gfile.GFile(save_path, mode='w') as writer:
        writer.write(json.dumps(raw_data, ensure_ascii=False, indent=2) + '\n')
    writer.close()


if __name__ == '__main__':
    generate_tfrecord_from_json_file(
        input_file_path='datasets/version_3/inference/valid.json',
        vocab_file_path='../bert-base-chinese/vocab.txt',
        tfrecord_save_path='datasets/version_3/inference/tfrecords/valid.tfrecord',
        meta_save_path='datasets/version_3/inference/meta/valid_meta.json',
        features_save_path='datasets/version_3/inference/features/valid_features.pkl',
        max_seq_len=200,
        is_train=False
    )

    # convert_last_step_results_for_train(
    #     results_path='../naive_mrc_task/inference_results/version_3/last_version_1/second/postprocessed/valid_results.json',
    #     save_path='datasets/version_3/train/valid.json'
    # )

    # convert_last_step_results_for_infer(
    #     results_path='../naive_mrc_task/inference_results/version_3/last_version_1/second/postprocessed/valid_results.json',
    #     save_path='datasets/version_3/inference/valid.json'
    # )

    # convert_last_step_results_for_valid(
    #     results_path='../naive_mrc_task/inference_results/last_task/use_version_2/second/postprocessed/'
    #                  'valid_results.json',
    #     save_path='datasets/from_last_version_2/raw/for_valid/valid.json'
    # )

    # postprocess_results(
    #     raw_data_path='datasets/from_last_version_1/raw/inference/valid.json',
    #     features_path='datasets/from_last_version_1/features/inference/valid_features.pkl',
    #     results_path='inference_results/last_version_1/raw/valid_results.json',
    #     save_path='inference_results/last_version_1/postprocessed/valid_results.json'
    # )

    pass
