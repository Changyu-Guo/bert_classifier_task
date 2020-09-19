# -*- coding: utf - 8 -*-

import collections
from data_processors.commom import extract_examples_from_init_train

init_train_txt_path = '../datasets/raw_datasets/init-train.txt'

init_train_examples = extract_examples_from_init_train(init_train_txt_path)

multi_answer_count = collections.defaultdict(lambda: 0)
total_multi_answer_relation_num = 0
total_single_answer_relation_num = 0

for index, init_train_example in enumerate(init_train_examples):

    sros = init_train_example.sros

    relation_number = collections.defaultdict(lambda: 0)

    multi_answer_relation_num = 0
    single_answer_relation_num = 0

    all_relations_set = set()
    extra_relations_set = set()

    for sro in sros:
        relation = sro['relation']
        relation_number[relation] += 1

        all_relations_set.add(relation)

        if relation in all_relations_set:
            extra_relations_set.add(relation)
            multi_answer_relation_num += 1
        else:
            single_answer_relation_num += 1

    multi_answer_relation_num += len(extra_relations_set)
    single_answer_relation_num -= len(extra_relations_set)

    total_multi_answer_relation_num += multi_answer_relation_num
    total_single_answer_relation_num += single_answer_relation_num

    for key, value in relation_number.items():
        multi_answer_count[value] += 1

for key, value in multi_answer_count.items():
    print(key, value)

total_relations = total_single_answer_relation_num + total_multi_answer_relation_num
print('total relations: ', total_relations)
print('multi answer relations: ', total_multi_answer_relation_num)
print('multi answer relations percent: ', total_multi_answer_relation_num / total_relations)

# 1 -> 14415
# 2 -> 698
# 3 -> 140

# 额外答案占比 6%
