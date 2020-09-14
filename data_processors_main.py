# -*- coding: utf - 8 -*-

from data_processors import mrc_data_processor
from data_processors import bi_cls_data_processor_s1

if __name__ == '__main__':
    bi_cls_data_processor_s1.read_examples_from_init_train(
        'datasets/raw_datasets/init-train-train.json'
    )
