# -*- coding: utf - 8 -*-

import json
from collections import Counter

INFERENCE_RESULT_PATH = 'inference_results/version_4/last_version_1/second/postprocessed/filtered_valid_results.json'

with open(INFERENCE_RESULT_PATH, mode='r', encoding='utf-8') as reader:
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

P_TPS = []
R_TPS = []
FPS = []
FNS = []

for paragraph in paragraphs:

    origin_sros = paragraph['origin_sros']
    pred_sros = paragraph['pred_sros']

    origin_relation_to_sro_dict = dict()
    for origin_sro in origin_sros:
        origin_relation_to_sro_dict[origin_sro['relation']] = origin_sro

    pred_relation_to_sro_dict = dict()
    for pred_sro in pred_sros:
        pred_relation_to_sro_dict[pred_sro['relation']] = pred_sro

    origin_relations = [origin_sro['relation'] for origin_sro in origin_sros]
    pred_relations = [pred_sro['relation'] for pred_sro in pred_sros]

    all_relations_set = set(origin_relations + pred_relations)

    P_TP = 0  # 正 -> 正
    R_TP = 0
    TN = 0  # 负 -> 负
    FP = 0  # 负 -> 正
    FN = 0  # 正 -> 负

    # 遍历所有的 relation
    for relation in all_relations_set:

        # true positive
        if relation in origin_relations and relation in pred_relations:
            # 计算字符级别的相关性
            origin_sro = origin_relation_to_sro_dict[relation]
            pred_sro = pred_relation_to_sro_dict[relation]
            origin_triad = origin_sro['subject'] + origin_sro['relation'] + origin_sro['object']
            pred_triad = pred_sro['subject'] + pred_sro['relation'] + pred_sro['object']
            common = Counter(pred_triad) & Counter(origin_triad)
            num_same = sum(common.values())
            if num_same == 0:
                P_TP += 0
                R_TP += 0
            else:
                P_TP += (1.0 * num_same / len(pred_triad))
                R_TP += (1.0 * num_same / len(origin_triad))

        # false negative
        if relation in origin_relations and relation not in pred_relations:
            FN += 1

        # false positive
        if relation not in origin_relations and relation in pred_relations:
            FP += 1

    if P_TP + FP == 0:
        precision = 0
    else:
        precision = P_TP / (P_TP + FP)

    if R_TP + FN == 0:
        recall = 0
    else:
        recall = R_TP / (R_TP + FN)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    P_TPS.append(P_TP)
    R_TPS.append(R_TP)
    FPS.append(FP)
    FNS.append(FN)

avg_P_TP = sum(P_TPS) / len(P_TPS)
avg_R_TP = sum(R_TPS) / len(R_TPS)
avg_FP = sum(FPS) / len(FPS)
avg_FN = sum(FNS) / len(FNS)

avg_precision = avg_P_TP / (avg_P_TP + avg_FP)
avg_recall = avg_R_TP / (avg_R_TP + avg_FN)
avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

print('precision', avg_precision)
print('recall', avg_recall)
print('f1', avg_f1)
