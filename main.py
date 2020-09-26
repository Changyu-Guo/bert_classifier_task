# -*- coding: utf - 8 -*-

from multi_turn_mrc_cls_task import main

if __name__ == '__main__':
    task = main.main()
    task.predict_tfrecord(
        'datasets/tfrecords/valid.tfrecord',
        'infer_results/valid_result.json'
    )

