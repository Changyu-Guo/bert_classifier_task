# -*- coding: utf - 8 -*-
"""
    Notice: Bert Version - chinese-bert-base
"""

import os
import json
import collections
import tensorflow as tf
from data_processor import load_init_train_txt
from data_processor import extract_relations_from_init_train_table
from utils import get_label_to_id_map
from squad_processor import read_squad_examples

relation_questions_txt_path = 'datasets/raw_datasets/relation_questions.txt'
vocab_filepath = './vocab.txt'
qa_examples_save_path = 'datasets/preprocessed_datasets/qa_examples.json'
cls_examples_save_path = 'datasets/preprocessed_datasets/cls_examples.json'


class InitTrainExample:
    """
        用于结构化 init-train.txt 中的数据
    """
    def __init__(self, text, relations):
        self.text = text
        self.relations = relations  # list of dict


class RelationQuestionsExample:
    """
        用于结构化 relation_questions.txt 中的数据
    """
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


def load_relation_questions_txt(filepath=relation_questions_txt_path):
    if not tf.io.gfile.exists(filepath):
        raise ValueError('Relation questions txt file {} not found'.format(filepath))
    with tf.io.gfile.GFile(filepath, mode='r') as reader:
        return reader.readlines()


def extract_examples_from_init_train():
    """
        Init Train Example
        {
            'text': '',
            relations: [{'subject': '', 'relation': '', 'object': ''}]
        }
    """
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


def extract_examples_from_relation_questions():
    """
        Relation Questions Example
        {
            'relation_id': '',
            'relation_name': '',
            'relation_question_a': '',
            'relation_question_b': '',
            'relation_question_c': ''
        }
    """
    relation_questions = load_relation_questions_txt()
    _, relations, _, _ = extract_relations_from_init_train_table()
    relation_to_id_map = get_label_to_id_map(relations)

    # 一个 relation 对应一个 dict
    # 理论上应该有 53 个 relation 对应 53 个 dict
    relation_questions_dict = collections.OrderedDict()
    for index in range(0, len(relations)):
        cur_index = index * 5

        relation = relation_questions[cur_index][1:-2].split(',')[1].split(':')[1]
        relation = relation.replace('"', '')

        relation_id = relation_to_id_map[relation]

        question_a = relation_questions[cur_index + 1].split('：')[1].strip()
        question_a = question_a.replace('？', '')

        question_b = relation_questions[cur_index + 2].split('：')[1].strip()
        question_b = question_b.replace('？', '')

        question_c = relation_questions[cur_index + 3].split('：')[1].strip()
        question_c = question_c.replace('？', '')

        relation_questions_dict[relation] = RelationQuestionsExample(
            relation_id, relation,
            question_a, question_b, question_c
        )

    return relation_questions_dict


def convert_example_to_squad_json_format(
        qa_examples_path=qa_examples_save_path,
        cls_examples_path=cls_examples_save_path
):
    init_train_examples = extract_examples_from_init_train()
    relation_questions_dict = extract_examples_from_relation_questions()

    # 存储用于分类的数据
    cls_examples = []

    # 存储用于阅读理解的数据
    squad_json = {
        'data': [
            {
                'title': 'temp title',
                'paragraphs': []
            }
        ]
    }
    _id = 0
    for item in init_train_examples:
        text = item.text
        relations = item.relations

        squad_json_item = {
            'context': text,
            'qas': []
        }
        for relation in relations:
            _subject = relation['subject']
            _object = relation['object']
            _relation = relation['relation']

            _subject_start_position = text.find(_subject)
            _object_start_position = text.find(_object)

            relation_question = relation_questions_dict[_relation]

            relation_question_a = relation_question.relation_question_a
            relation_question_b = relation_question.relation_question_b.replace('subject', _subject)
            relation_question_c = relation_question.relation_question_c.replace(
                'subject', _subject
            ).replace(
                'object', _object
            )

            # to squad json format
            # question 1, for mrc
            qas_item = {
                'question': relation_question_a,
                'answers': [
                    {
                        'answer_start': _subject_start_position,
                        'text': _subject
                    }
                ],
                'id': _id
            }
            squad_json_item['qas'].append(qas_item)
            _id += 1

            # question 2, for mrc
            qas_item = {
                'question': relation_question_b,
                'answers': [
                    {
                        'answer_start': _object_start_position,
                        'text': _object
                    }
                ],
                'id': _id
            }
            squad_json_item['qas'].append(qas_item)
            _id += 1

            squad_json['data'][0]['paragraphs'].append(squad_json_item)

            # question 3, for classification
            cls_item = {
                'text': text,
                'question': relation_question_c,
                'is_valid': True
            }
            cls_examples.append(cls_item)

            # TODO 考虑根据当前正样本随机加入一些负样本

            # logging info
            if _id % 1000 == 0:
                print(_id)

    # 将 mrc 数据写入 json 文件
    with tf.io.gfile.GFile(qa_examples_path, 'w') as writer:
        writer.write(json.dumps(squad_json, ensure_ascii=False, indent=2))
    writer.close()

    # 将 cls 数据写入 json 文件
    with tf.io.gfile.GFile(cls_examples_path, 'w') as writer:
        writer.write(json.dumps(cls_examples, ensure_ascii=False, indent=2))
    writer.close()


if __name__ == '__main__':
    convert_example_to_squad_json_format()
