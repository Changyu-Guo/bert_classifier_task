# -*- coding: utf - 8 -*-

from data_processors import mrc_data_processor
from data_processors import bi_cls_data_processor_s1
from data_processors import commom

if __name__ == '__main__':
    bi_cls_data_processor_s1.read_valid_examples_from_init_train(
        'datasets/raw_datasets/init-train-valid.json'
    )
