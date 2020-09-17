# -*- coding: utf - 8 -*-

import json
import tensorflow as tf
from data_processors.commom import extract_examples_dict_from_relation_questions

# bi cls s1 inference result
bi_cls_s1_train_results_path = 'inference_results/bi_cls_s1_results/in_use/train_results.json'
bi_cls_s1_valid_results_path = 'inference_results/bi_cls_s1_results/in_use/valid_results.json'

# for mrc first step
train_data_before_first_step_save_path = \
    'datasets/preprocessed_datasets/before_mrc_first_step/in_use/first_step_train.json'
valid_data_before_first_step_save_path = \
    'datasets/preprocessed_datasets/before_mrc_first_step/in_use/first_step_valid.json'


# 对于第一步骤抽取 relation 的推断结果
# 将其转为 squad 数据类型
def convert_inference_results_for_mrc_first_step(
        inference_results_path, convert_results_save_path
):
    # 获取所有的 relation question
    relation_questions_dict = extract_examples_dict_from_relation_questions()

    # 加载上一步骤的推断结果
    with tf.io.gfile.GFile(inference_results_path, mode='r') as reader:
        inference_results = json.load(reader)
    reader.close()

    # TODO: 改为函数调用
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
        origin_sros = item['sros']
        pred_sros = item['pred_sros']

        # TODO: 改为函数调用
        squad_json_item = {
            'context': text,
            'qas': [],
            'origin_sros': origin_sros,
            'pred_sros': pred_sros
        }

        for sro in pred_sros:
            relation_question = relation_questions_dict[sro['relation']]
            question_a = relation_question.question_a

            qas_item = {
                'question': question_a,
                'relation': sro['relation'],
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


def convert_train_for_first_step():
    convert_inference_results_for_mrc_first_step(
        inference_results_path=bi_cls_s1_train_results_path,
        convert_results_save_path=train_data_before_first_step_save_path
    )


def convert_valid_for_first_step():
    convert_inference_results_for_mrc_first_step(
        inference_results_path=bi_cls_s1_valid_results_path,
        convert_results_save_path=valid_data_before_first_step_save_path
    )


if __name__ == '__main__':
    convert_valid_for_first_step()
