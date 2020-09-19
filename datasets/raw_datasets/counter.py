# -*- coding: utf - 8 -*-

import json
import tensorflow as tf


def count_init_train_sro_num():
    with tf.io.gfile.GFile('init-train.txt', mode='r') as reader:
        init_train_data = json.load(reader)
    reader.close()
    pass


def count_init_train_train_sro_num():
    pass


def count_init_train_valid_sro_num():
    pass


if __name__ == '__main__':
    pass
