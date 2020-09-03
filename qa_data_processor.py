# -*- coding: utf - 8 -*-
"""
    Notice: Bert Version - chinese-bert-base
"""

import os
import json
import collections
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from data_processor import load_init_train_txt
from data_processor import extract_relations_from_init_train_table
from utils import get_label_to_id_map

relation_questions_txt_path = './datasets/relation_questions.txt'
tfrecord_save_path = './datasets/relation_qas.tfrecord'
desc_json_save_path = './datasets/relation_qas_desc.json'
vocab_filepath = './vocab.txt'


class InitTrainExample:
    def __init__(self, text, relations):
        self.text = text
        self.relations = relations


class RelationQuestionsExample:
    def __init__(
            self,
            relation_id,
            relation_name,
            relation_question_a,
            relation_question_b,
            relation_question_c,
    ):
        self.relation_id = relation_id
        self.relation_name = relation_name
        self.relation_question_a = relation_question_a
        self.relation_question_b = relation_question_b
        self.relation_question_c = relation_question_c


class QAExample:
    def __init__(self, text, question, answer_text, start_position):
        self.text = text
        self.question = question
        self.answer_text = answer_text
        self.start_position = start_position


class CLSExample:
    def __init__(self, text, question, is_valid):
        self.text = text
        self.question = question
        self.is_valid = is_valid


class Feature:
    pass


class FeaturesWriter:
    pass


def load_relation_questions_txt(filepath=relation_questions_txt_path):
    if not tf.io.gfile.exists(filepath):
        raise ValueError('Relation questions txt file {} not found'.format(filepath))
    with tf.io.gfile.GFile(filepath, mode='r') as reader:
        return reader.readlines()


def extract_examples_from_init_train():
    init_train = load_init_train_txt()
    examples = []
    for item in init_train:
        item = json.loads(item)
        text = item['text'].strip()
        sro_list = item['sro_list']
        relations = [
            {
                'subject': sro['subject'],
                'relation': sro['relation'],
                'object': sro['object']
            } for sro in sro_list
        ]
        example = InitTrainExample(text, relations)
        examples.append(example)

    return examples


def extract_questions_from_relation_qa():
    relation_questions = load_relation_questions_txt()
    _, relations, _, _ = extract_relations_from_init_train_table()
    relation_to_id_map = get_label_to_id_map(relations)

    relation_questions_dict = collections.OrderedDict()
    for index in range(0, len(relations)):
        cur_index = index * 5

        relation = relation_questions[cur_index][1:-2].split(',')[1].split(':')[1]
        relation = relation.replace('"', '')

        relation_id = relation_to_id_map[relation]

        question_a = relation_questions[cur_index + 1].split('：')[1]
        question_a = question_a.replace('？', '')

        question_b = relation_questions[cur_index + 2].split('：')[1]
        question_b = question_b.replace('？', '')

        question_c = relation_questions[cur_index + 3].split('：')[1]
        question_c = question_c.replace('？', '')

        relation_questions_dict[relation] = RelationQuestionsExample(
            relation_id, relation,
            question_a, question_b, question_c
        )

    return relation_questions_dict


def extract_examples_from_relation_questions():
    """
        抽取思路：
            1. 从 init-train.txt 中获取 text, relation, relation 对应的 answer
                具体的：将 init-train.txt 和 relation-question.txt 都弄成 object, 然后使用 dict 循环寻找对应关系就行了
            2. 从 relation-questions.txt 中抽取 relation 对应的答案
            3. 根据 relation, 将 init-train.txt 和 relation-questions.txt 联系起来
            4. 形成 text:question:answer:start_position:end_position 问答数据
            5. 形成 text:question:answer 分类数据
    """
    init_train_examples = extract_examples_from_init_train()
    relation_question_examples = extract_questions_from_relation_qa()

    qa_examples = []
    cls_examples = []

    for item in init_train_examples:
        text = item.text
        relations = item.relations

        for relation in relations:
            _subject = relation['subject']
            _object = relation['object']
            _relation = relation['relation']

            _subject_start_position = text.find(_subject)
            _object_start_position = text.find(_object)

            relation_question = relation_question_examples[_relation]

            relation_question_a = relation_question.relation_question_a
            relation_question_b = relation_question.relation_question_b.replace('subject', _subject)
            relation_question_c = relation_question.relation_question_c.replace(
                'subject', _subject
            ).replace(
                'object', _object
            )

            # question 1
            qa_example = QAExample(
                text,
                relation_question_a,
                _subject,
                _subject_start_position
            )
            qa_examples.append(qa_example)

            # question 2
            qa_example = QAExample(
                text,
                relation_question_b,
                _object,
                _object_start_position
            )
            qa_examples.append(qa_example)

            # question 3, for classification
            cls_example = CLSExample(
                text,
                relation_question_c,
                is_valid=True
            )
            cls_examples.append(cls_example)

            # TODO 考虑根据当前正样本随机加入一些负样本

    return qa_examples, cls_examples


def convert_qa_example_to_feature(examples):
    pass


def convert_cls_example_to_feature(examples):
    pass


def convert_qa_and_cls_example_to_feature():
    qa_examples, cls_examples = extract_examples_from_relation_questions()
    convert_qa_example_to_feature(qa_examples)
    convert_cls_example_to_feature(cls_examples)


if __name__ == '__main__':
    convert_qa_and_cls_example_to_feature()
