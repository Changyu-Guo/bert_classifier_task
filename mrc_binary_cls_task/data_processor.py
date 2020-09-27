# -*- coding: utf - 8 -*-

import gzip
import json
import pickle
import collections
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from common_data_utils import extract_examples_dict_from_relation_questions


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


def read_examples_from_last_step_results(last_step_result_path, is_training):
    relation_questions_dict = extract_examples_dict_from_relation_questions(
        init_train_table_path='../common-datasets/init-train-table.txt',
        relation_questions_path='../common-datasets/relation_questions.txt'
    )
    with tf.io.gfile.GFile(last_step_result_path, mode='r') as reader:
        results = json.load(reader)
    reader.close()

    examples = []
    paragraphs = results['data'][0]['paragraphs']

    for paragraph_index, paragraph in enumerate(paragraphs):
        context = paragraph['context']
        origin_sros = paragraph['origin_sros']
        pred_sros = paragraph['pred_sros']
        origin_three_tuples = []

        # 构造正样本
        for sro_index, sro in enumerate(origin_sros):
            r = sro['relation']
            s = sro['subject']
            o = sro['object']
            origin_three_tuples.append(s + r + o)

            relation_questions = relation_questions_dict[r]
            question_c = relation_questions.question_c
            question_c = question_c.replace('subject', s).replace('object', o)

            example = BiCLSExample(
                paragraph_index=paragraph_index,
                sro_index=sro_index,
                text=context,
                question=question_c,
                is_valid=1
            )
            examples.append(example)

        # 构造负样本
        for sro_index, sro in enumerate(pred_sros):
            if not sro.get('object', False):
                continue

            s = sro['subject']
            r = sro['relation']
            o = sro['object']

            three_tuple = s + r + o
            # 如果当前样本预测正确，则不构造当前样本为负样本
            if three_tuple in origin_three_tuples:
                continue

            relation_questions = relation_questions_dict[r]
            question_c = relation_questions.question_c
            question_c = question_c.replace('subject', s).replace('object', o)

            example = BiCLSExample(
                paragraph_index=paragraph_index,
                sro_index=sro_index,
                text=context,
                question=question_c,
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
    examples = read_examples_from_last_step_results(input_file_path, is_training=is_train)
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

    examples = read_examples_from_last_step_results(
        last_step_result_path=raw_data_path,
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
        if prob >= 0.5:
            paragraphs[paragraph_index]['pred_sros'][sro_index]['is_valid'] = 1

    raw_data['data'][0]['paragraphs'] = paragraphs

    with tf.io.gfile.GFile(save_path, mode='w') as writer:
        writer.write(json.dumps(raw_data, ensure_ascii=False, indent=2) + '\n')
    writer.close()


if __name__ == '__main__':
    generate_tfrecord_from_json_file(
        input_file_path='../naive_mrc_task/infer_results/last_task/use_version_2/'
                        'second_step/postprocessed/valid_results.json',
        vocab_file_path='../vocabs/bert-base-chinese-vocab.txt',
        tfrecord_save_path='datasets/from_last_version_2/tfrecords/for_train/valid.tfrecord',
        meta_save_path='datasets/from_last_version_2/tfrecords/for_train/valid_meta.json',
        features_save_path='datasets/from_last_version_2/features/for_train/valid_features.pkl',
        max_seq_len=165,
        is_train=True
    )
