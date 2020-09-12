# -*- coding: utf - 8 -*-
"""
    Notice: Bert Version - chinese-bert-base

    Convert MRC data to squad format.
"""

import json
import collections
import tensorflow as tf
from data_processors.multi_label_cls_data_processor import load_init_train_txt
from data_processors.multi_label_cls_data_processor import extract_relations_from_init_train_table
from utils.data_utils import get_label_to_id_map

vocab_filepath = 'vocabs/bert-base-chinese-vocab.txt'

# all relation questions
relation_questions_txt_path = 'datasets/raw_datasets/relation_questions.txt'

# mrc task
mrc_train_save_path = 'datasets/preprocessed_datasets/mrc_train.json'
mrc_valid_save_path = 'datasets/preprocessed_datasets/mrc_valid.json'

# multi label cls step inference result
multi_label_cls_train_results_path = 'inference_results/multi_label_cls_results/in_use/train_results.json'
multi_label_cls_valid_results_path = 'inference_results/multi_label_cls_results/in_use/valid_results.json'

# first step json file
first_step_train_save_path = 'datasets/preprocessed_datasets/first_step_train.json'
first_step_valid_save_path = 'datasets/preprocessed_datasets/first_step_valid.json'

# first step inference results
first_step_inference_train_save_path = 'inference_results/mrc_results/in_use/first_step/train_results.json'
first_step_inference_valid_save_path = 'inference_results/mrc_results/in_use/second_step/valid_results.json'

# second step json file
second_step_train_save_path = 'datasets/preprocessed_datasets/second_step_train.json'
second_step_valid_save_path = 'datasets/preprocessed_datasets/second_step_valid.json'

# second step inference results
second_step_inference_train_save_path = 'inference_results/mrc_results/in_use/second_step/train_results.json'
second_step_inference_valid_save_path = 'inference_results/mrc_results/in_use/second_step/valid_results.json'


class InitTrainExample:
    """
        用于结构化 init-train.txt 中的数据

        text: str
        relations: [{'relation': '', 'subject': '', 'object': ''}]
    """
    def __init__(self, text, relations):
        self.text = text
        self.relations = relations  # list of dict


class RelationQuestionsExample:
    """
        用于结构化 relation_questions.txt 中的数据

        relation_id: int
        relation_name: str
        relation_question_a: str
        relation_question_b: str
        relation_question_c: str
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

        'text': str,
        relations: [{'relation': '', 'subject': '', 'object': ''}]
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


def extract_examples_from_relation_questions():
    """
        Relation Questions Example

            'relation_id': str,
            'relation_name': str,
            'relation_question_a': str,
            'relation_question_b': str,
            'relation_question_c': str
    """
    relation_questions = load_relation_questions_txt()
    _, relations, _, _ = extract_relations_from_init_train_table()
    relation_to_id_map = get_label_to_id_map(relations)

    # 一个 relation 对应一个 dict
    # 理论上应该有 53 个 relation 对应 53 个 dict
    relation_questions_dict = collections.OrderedDict()
    for index in range(0, len(relations)):
        cur_index = index * 5

        relation_name = relation_questions[cur_index][1:-2].split(',')[1].split(':')[1].strip()
        relation_name = relation_name.replace('"', '')

        relation_id = relation_to_id_map[relation_name]

        question_a = relation_questions[cur_index + 1].split('：')[1].strip()
        question_a = question_a.replace('？', '')

        question_b = relation_questions[cur_index + 2].split('：')[1].strip()
        question_b = question_b.replace('？', '')

        question_c = relation_questions[cur_index + 3].split('：')[1].strip()
        question_c = question_c.replace('？', '')

        relation_questions_dict[relation_name] = RelationQuestionsExample(
            relation_id, relation_name,
            question_a, question_b, question_c
        )

    return relation_questions_dict


def convert_example_to_squad_json_format(mrc_train_path, mrc_valid_path, cls_train_path, cls_valid_path):
    init_train_examples = extract_examples_from_init_train()
    relation_questions_dict = extract_examples_from_relation_questions()

    # data for mrc
    train_squad_json = {
        'data': [
            {
                'title': 'train data',
                'paragraphs': []
            }
        ]
    }
    valid_squad_json = {
        'data': [
            {
                'title': 'eval data',
                'paragraphs': []
            }
        ]
    }
    # 为每一条 example 分配一个唯一的字符串 id
    _id = 0
    for item_index, item in enumerate(init_train_examples):
        text = item.text
        relations = item.relations

        # 针对当前文本，有若干个 问题 - 答案 对
        squad_json_item = {
            'context': text,
            'qas': []
        }
        for relation in relations:

            # sro item
            _subject = relation['subject']
            _object = relation['object']
            _relation = relation['relation']

            # answer start position
            _subject_start_position = text.find(_subject)
            _object_start_position = text.find(_object)

            # 当前 relation 对应的 question
            relation_question = relation_questions_dict[_relation]
            relation_question_a = relation_question.relation_question_a
            relation_question_b = relation_question.relation_question_b.replace('subject', _subject)

            # to squad json format
            # question 1, for mrc
            qas_item = {
                'question': relation_question_a,
                'answers': [
                    {
                        'text': _subject,
                        'answer_start': _subject_start_position
                    }
                ],
                'id': 'id_' + str(_id)
            }
            squad_json_item['qas'].append(qas_item)
            _id += 1

            # question 2, for mrc
            qas_item = {
                'question': relation_question_b,
                'answers': [
                    {
                        'text': _object,
                        'answer_start': _object_start_position
                    }
                ],
                'id': 'id_' + str(_id)
            }
            squad_json_item['qas'].append(qas_item)
            _id += 1

        # 当前 item 被分到验证集
        if (item_index + 1) % 10 == 0:
            valid_squad_json['data'][0]['paragraphs'].append(squad_json_item)
        else:
            # 当前 item 被分到训练集
            train_squad_json['data'][0]['paragraphs'].append(squad_json_item)

        if (item_index + 1) % 1000 == 0:
            print(item_index + 1)

    # 将 mrc 数据写入 json 文件
    with tf.io.gfile.GFile(mrc_train_path, 'w') as writer:
        writer.write(json.dumps(train_squad_json, ensure_ascii=False, indent=2))
    writer.close()
    with tf.io.gfile.GFile(mrc_valid_path, 'w') as writer:
        writer.write(json.dumps(valid_squad_json, ensure_ascii=False, indent=2))
    writer.close()


def convert_inference_results_for_first_step(inference_results_path, convert_results_save_path):
    relation_questions_dict = extract_examples_from_relation_questions()
    with tf.io.gfile.GFile(inference_results_path, mode='r') as reader:
        inference_results = json.load(reader)
    reader.close()

    squad_json = {
        'data': [
            {
                'title': 'first step data',
                'paragraphs': []
            }
        ]
    }

    _id = 0
    for item_index, item in enumerate(inference_results):
        text = item['text']
        origin_relations = item['origin_relations']
        pred_relations = item['pred_relations']

        squad_json_item = {
            'context': text,
            'qas': [],
            'origin_relations': origin_relations,
            'pred_relations': pred_relations
        }

        for relation in pred_relations:
            relation_question = relation_questions_dict[relation['relation']]
            relation_question_a = relation_question.relation_question_a

            qas_item = {
                'question': relation_question_a,
                'id': 'id_' + str(_id)
            }
            squad_json_item['qas'].append(qas_item)
            _id += 1

        squad_json['data'][0]['paragraphs'].append(squad_json_item)

        if (item_index + 1) % 1000 == 0:
            print(item_index + 1)

    with tf.io.gfile.GFile(convert_results_save_path, mode='w') as writer:
        writer.write(json.dumps(squad_json, ensure_ascii=False, indent=2))
    writer.close()


def convert_inference_results_for_second_step():
    pass


def mrc_data_processor_main():
    convert_inference_results_for_first_step(
        inference_results_path=multi_label_cls_valid_results_path,
        convert_results_save_path=first_step_valid_save_path
    )


if __name__ == '__main__':
    pass
