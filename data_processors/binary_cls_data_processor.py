# -*- coding: utf - 8 -*-

import json
import collections
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from data_processors.mrc_data_processor import extract_examples_from_relation_questions

inference_train_path = '../inference_results/mrc_results/in_use/second_step/train_results.json'
inference_valid_path = '../inference_results/mrc_results/in_use/second_step/valid_results.json'


class BiCLSExample:
    def __init__(self, text, question, is_valid):
        self.text = text
        self.question = question
        self.is_valid = is_valid


class BiCLSFeature:
    def __init__(
            self,
            inputs_ids,
            inputs_mask,
            segment_ids,
            is_valid
    ):
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
        features['inputs_ids'] = create_int_feature(feature.inputs_ids)
        features['inputs_mask'] = create_int_feature(feature.inputs_mask)
        features['segment_ids'] = create_int_feature(feature.segment_ids)
        features['is_valid'] = create_int_feature([feature.is_valid])

        example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(example.SerializeToString())

    def close(self):
        self._writer.close()


def read_examples_from_mrc_inference_results(filepath):
    relation_questions_dict = extract_examples_from_relation_questions()
    with tf.io.gfile.GFile(filepath, mode='r') as reader:
        input_data = json.load(reader)['data'][0]

    examples = []
    paragraphs = input_data['paragraphs']

    for paragraph in paragraphs:
        context = paragraph['context']
        origin_relations = paragraph['origin_relations']
        pred_relations = paragraph['pred_relations']
        origin_three_tuples = []

        for relation in origin_relations:
            _relation = relation['relation']
            _subject = relation['subject']
            _object = relation['object']
            origin_three_tuples.append(_subject + _relation + _object)

            relation_questions = relation_questions_dict[_relation]
            relation_question_c = relation_questions.relation_question_c
            relation_question_c = relation_question_c.replace('subject', _subject).replace('object', _object)

            example = BiCLSExample(
                text=context,
                question=relation_question_c,
                is_valid=1
            )
            examples.append(example)

        for relation in pred_relations:
            _relation = relation['relation']
            _subject = relation['subject']
            _object = relation['object']

            three_tuple = _subject + _relation + _object
            if three_tuple in origin_three_tuples:
                continue

            relation_questions = relation_questions_dict[_relation]
            relation_question_c = relation_questions.relation_question_c
            relation_question_c = relation_question_c.replace('subject', _subject).replace('object', _object)

            example = BiCLSExample(
                text=context,
                question=relation_question_c,
                is_valid=0
            )
            examples.append(example)

    return examples


def convert_examples_to_features(
        examples, vocab_file_path, max_seq_len, output_fn
):
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file_path)
    tokenizer.enable_padding(length=max_seq_len)
    tokenizer.enable_truncation(max_seq_len)

    for example_index, example in enumerate(examples):
        text = example.text
        question = example.question
        is_valid = example.is_valid

        tokenizer_outputs = tokenizer.encode(question, text)

        feature = BiCLSFeature(
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


def binary_cls_data_processor_main():
    vocab_file_path = 'vocabs/bert-base-chinese-vocab.txt'
    examples = read_examples_from_mrc_inference_results(
        'inference_results/mrc_results/in_use/second_step/valid_results.json'
    )
    convert_examples_to_features(
        examples,
        vocab_file_path=vocab_file_path,
        max_seq_len=165,
        output_fn=None
    )


if __name__ == '__main__':
    pass
