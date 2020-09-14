# -*- coding: utf - 8 -*-

import json
import collections
import tensorflow as tf

INIT_TRAIN_TABLE_TXT_PATH = 'datasets/raw_datasets/init-train-table.txt'
INIT_TRAIN_TXT_PATH = 'datasets/raw_datasets/init-train.txt'

INIT_TRAIN_TRAIN_SAVE_PATH = 'datasets/raw_datasets/init-train-train.json'
INIT_TRAIN_VALID_SAVE_PATH = 'datasets/raw_datasets/init-train-valid.json'

RELATION_QUESTIONS_TXT_PATH = 'datasets/raw_datasets/relation_questions.txt'


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
            relation_question_c
    ):
        self.relation_id = relation_id
        self.relation_name = relation_name
        self.relation_question_a = relation_question_a
        self.relation_question_b = relation_question_b
        self.relation_question_c = relation_question_c


def load_init_train_table_txt(init_train_table_txt_path=INIT_TRAIN_TABLE_TXT_PATH):
    if not tf.io.gfile.exists(init_train_table_txt_path):
        raise ValueError('Init train table txt file {} not found.'.format(init_train_table_txt_path))
    with tf.io.gfile.GFile(init_train_table_txt_path, mode='r') as reader:
        return reader.readlines()


def load_init_train_txt(init_train_txt_path=INIT_TRAIN_TXT_PATH):
    if not tf.io.gfile.exists(init_train_txt_path):
        raise ValueError('Init train txt file {} not found.'.format(init_train_txt_path))
    with tf.io.gfile.GFile(init_train_txt_path, mode='r') as reader:
        return reader.readlines()


def load_relation_questions_txt(relation_questions_txt_path=RELATION_QUESTIONS_TXT_PATH):
    if not tf.io.gfile.exists(relation_questions_txt_path):
        raise ValueError('Relation questions txt file {} not found'.format(relation_questions_txt_path))
    with tf.io.gfile.GFile(relation_questions_txt_path, mode='r') as reader:
        return reader.readlines()


def extract_relations_from_init_train_table(init_train_table_path=INIT_TRAIN_TABLE_TXT_PATH):
    init_train_table = load_init_train_table_txt(init_train_table_path)

    subjects, relations, objects, combined_str = [], [], [], []

    for item in init_train_table:

        # 去掉 '{}'
        item = item.strip()[1:-2]

        s, r, o = item.split(',')
        identity = ''

        for ls, it in zip((subjects, relations, objects), (s, r, o)):
            it = it.split(':')[1].replace('"', '')
            ls.append(it)
            identity += it
        combined_str.append(identity)

    relations_set = set(relations)
    combined_str_set = set(combined_str)

    assert len(relations_set) == len(relations) == len(init_train_table) == len(combined_str_set)

    return subjects, relations, objects, combined_str


def extract_examples_from_init_train(init_train_path=INIT_TRAIN_TXT_PATH):
    """
        Extract Init Train Example
    """
    init_train = load_init_train_txt(init_train_path)
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


def split_init_train_data(
        init_train_path=INIT_TRAIN_TXT_PATH,
        split_train_save_path=INIT_TRAIN_TRAIN_SAVE_PATH,
        split_valid_save_path=INIT_TRAIN_VALID_SAVE_PATH,
        split_valid_ratio=0.1
):
    """
        从 Example 的层面切分数据集
        切分后的数据集保存的是 Example
    """
    split_valid_index = 1 / split_valid_ratio
    init_train_examples = extract_examples_from_init_train(init_train_path)
    split_train_examples = []
    split_valid_examples = []
    for index, example in enumerate(init_train_examples):
        item = {
            'text': example.text,
            'relations': example.relations
        }
        if (index + 1) % split_valid_index == 0:
            split_valid_examples.append(item)
        else:
            split_train_examples.append(item)

    with tf.io.gfile.GFile(split_train_save_path, mode='w') as writer:
        writer.write(json.dumps(split_train_examples, ensure_ascii=False, indent=2))
    writer.close()
    with tf.io.gfile.GFile(split_valid_save_path, mode='w') as writer:
        writer.write(json.dumps(split_valid_examples, ensure_ascii=False, indent=2))
    writer.close()


def read_init_train_train_examples(train_examples_path=INIT_TRAIN_TRAIN_SAVE_PATH):
    with tf.io.gfile.GFile(train_examples_path, mode='r') as reader:
        examples = json.load(reader)
    return examples


def read_init_train_valid_examples(valid_examples_path=INIT_TRAIN_VALID_SAVE_PATH):
    with tf.io.gfile.GFile(valid_examples_path, mode='r') as reader:
        examples = json.load(reader)
    return examples


def extract_examples_dict_from_relation_questions(
        init_train_table_path=INIT_TRAIN_TABLE_TXT_PATH,
        relation_questions_path=RELATION_QUESTIONS_TXT_PATH
):
    relation_questions = load_relation_questions_txt(relation_questions_path)
    _, relations, _, _ = extract_relations_from_init_train_table(init_train_table_path)

    relation_to_index_map = collections.OrderedDict()
    for index, item in enumerate(relations):
        relation_to_index_map[item] = index

    # 一个 relation 对应一个 dict
    # 理论上应该有 53 个 relation 对应 53 个 dict
    relation_questions_dict = collections.OrderedDict()
    for index in range(0, len(relations)):
        cur_index = index * 5

        relation_name = relation_questions[cur_index][1:-2].split(',')[1].split(':')[1].strip()
        relation_name = relation_name.replace('"', '')

        relation_id = relation_to_index_map[relation_name]

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


def get_squad_json_template(title):
    return {
        'data': [
            {
                'title': title,
                'paragraphs': []
            }
        ]
    }


def get_squad_json_paragraph_template(text, origin_relations):
    return {
        'context': text,
        'qas': [],
        # 存储当前 text 对应的所有 relation
        # 用在推断任务中
        'origin_relations': origin_relations
    }


def get_squad_json_qas_item_template(question, answers, qas_id):
    return {
        'question': question,
        'answers': answers,
        'id': qas_id
    }
