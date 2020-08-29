# -*- coding: utf - 8 -*-

import os
import json
import collections
import tensorflow as tf
from absl import logging


def get_label_to_id_map(labels):
    label_to_id_map = collections.OrderedDict()
    for i, v in enumerate(labels):
        label_to_id_map[v] = i

    return label_to_id_map


def labels_to_ids(label_to_id_map, labels):
    return [label_to_id_map[label] for label in labels]


def ids_to_vector(ids, _len):
    zeros = [0] * _len
    for _id in ids:
        zeros[_id] = 1
    return zeros
