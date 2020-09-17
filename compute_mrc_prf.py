# -*- coding: utf - 8 -*-

import json
import tensorflow as tf

INFERENCE_RESULTS_PATH = 'inference_results/mrc_results/in_use/second_step/valid_results.json'

with tf.io.gfile.GFile(INFERENCE_RESULTS_PATH, mode='r') as reader:
    results = json.load(reader)
reader.close()


results = results['data'][0]

paragraphs = results['paragraphs']

"""
    P = TP / (TP + FP)
    
    R = TP / (TP + FN)
    
    F = 2 * P * R / (P + R)
"""

precisions = []
recalls = []
f1s = []

TPS = []
FPS = []
FNS = []

for paragraph in paragraphs:
    origin_sros = paragraph['origin_sros']
    pred_sros = paragraph['pred_sros']

    all_relation_tuples_set = set()

    origin_relation_tuples = []
    for origin_sro in origin_sros:
        relation_tuple = origin_sro['subject'] + origin_sro['relation'] + origin_sro['object']
        origin_relation_tuples.append(relation_tuple)
        all_relation_tuples_set.add(relation_tuple)

    pred_relation_tuples = []
    for pred_sro in pred_sros:
        relation_tuple = pred_sro['subject'] + pred_sro['relation'] + pred_sro['object']
        pred_relation_tuples.append(relation_tuple)
        all_relation_tuples_set.add(relation_tuple)

    TP = 0  # 正 -> 正
    TN = 0  # 负 -> 负
    FP = 0  # 负 -> 正
    FN = 0  # 正 -> 负

    for relation_tuple in all_relation_tuples_set:

        # 正 -> 正
        if relation_tuple in origin_relation_tuples and relation_tuple in pred_relation_tuples:
            TP += 1

        # 正 -> 负
        if relation_tuple in origin_relation_tuples and relation_tuple not in pred_relation_tuples:
            FN += 1

        # 负 -> 正
        if relation_tuple not in origin_relation_tuples and relation_tuple in pred_relation_tuples:
            FP += 1

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    TPS.append(TP)
    FPS.append(FP)
    FNS.append(FN)


# final_precision = sum(precisions) / len(precisions)
# final_recall = sum(recalls) / len(recalls)
# final_f1 = sum(f1s) / len(f1s)
#
# print(final_precision)
# print(final_recall)
# print(final_f1)

avg_TP = sum(TPS) / len(TPS)
avg_FP = sum(FPS) / len(FPS)
avg_FN = sum(FNS) / len(FNS)

avg_precision = avg_TP / (avg_TP + avg_FP)
avg_recall = avg_TP / (avg_TP + avg_FN)
avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

print('precision', avg_precision)
print('recall', avg_recall)
print('f1', avg_f1)

