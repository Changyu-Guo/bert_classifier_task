# -*- coding: utf - 8 -*-

import json
import collections
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from data_processors.commom import extract_examples_dict_from_relation_questions


class BiCLSExample:
    def __init__(self, text, question, is_valid):
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
    def __init__(self, filename):
        self.filename = filename

        if tf.io.gfile.exists(filename):
            tf.io.gfile.remove(filename)

        options = tf.io.TFRecordOptions(compression_type='GZIP')
        self._writer = tf.io.TFRecordWriter(filename, options=options)
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


def read_examples_from_mrc_inference_results(filepath):
    relation_questions_dict = extract_examples_dict_from_relation_questions()
    with tf.io.gfile.GFile(filepath, mode='r') as reader:
        input_data = json.load(reader)['data'][0]

    examples = []
    paragraphs = input_data['paragraphs']

    for paragraph in paragraphs:
        context = paragraph['context']
        origin_sros = paragraph['origin_sros']
        pred_sros = paragraph['pred_sros']
        origin_three_tuples = []

        # 构造正样本
        for sro in origin_sros:
            _relation = sro['relation']
            _subject = sro['subject']
            _object = sro['object']
            origin_three_tuples.append(_subject + _relation + _object)

            relation_questions = relation_questions_dict[_relation]
            question_c = relation_questions.question_c
            question_c = question_c.replace('subject', _subject).replace('object', _object)

            example = BiCLSExample(
                text=context,
                question=question_c,
                is_valid=1
            )
            examples.append(example)

        # 构造负样本
        for sro in pred_sros:
            _relation = sro['relation']
            _subject = sro['subject']
            _object = sro['object']

            three_tuple = _subject + _relation + _object
            # 如果当前样本预测正确，则不构造当前样本为负样本
            if three_tuple in origin_three_tuples:
                continue

            relation_questions = relation_questions_dict[_relation]
            question_c = relation_questions.question_c
            question_c = question_c.replace('subject', _subject).replace('object', _object)

            example = BiCLSExample(
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
        output_file_path,
        max_seq_len=128
):
    examples = read_examples_from_mrc_inference_results(input_file_path)
    writer = FeatureWriter(filename=output_file_path)
    convert_examples_to_features(
        examples,
        vocab_file_path,
        max_seq_len,
        writer.process_feature
    )
    meta_data = {
        'task_type': 'bi_cls',
        'data_size': writer.total_features,
        'max_seq_len': max_seq_len
    }

    writer.close()

    return meta_data


if __name__ == '__main__':
    """
        该文件已经过审查
    """
    pass
