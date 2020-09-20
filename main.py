# -*- coding: utf - 8 -*-

from data_processors.mrc_data_processor import mrc_data_processor_main
from mrc_task import mrc_predict_main
from mrc_task import mrc_main

vocab_filepath = 'vocabs/bert-base-chinese-vocab.txt'

# all relation questions
relation_questions_txt_path = 'common-datasets/relation_questions.txt'

# mrc task
mrc_train_save_path = 'common-datasets/preprocessed_datasets/mrc_train.json'
mrc_valid_save_path = 'common-datasets/preprocessed_datasets/mrc_valid.json'

# multi label cls step inference result
multi_label_cls_train_results_path = 'inference_results/multi_label_cls_results/in_use/train_results.json'
multi_label_cls_valid_results_path = 'inference_results/multi_label_cls_results/in_use/valid_results.json'

# 第一步推断使用的输入数据位置
first_step_train_save_path = 'common-datasets/preprocessed_datasets/first_step_train.json'
first_step_valid_save_path = 'common-datasets/preprocessed_datasets/first_step_valid.json'

# 第一次推断的输出数据保存位置
first_step_inference_train_save_path = 'inference_results/mrc_results/in_use/first_step/train_results.json'
first_step_inference_valid_save_path = 'inference_results/mrc_results/in_use/second_step/valid_results.json'

# 第二次推断使用的输入数据位置
second_step_train_save_path = 'common-datasets/preprocessed_datasets/second_step_train.json'
second_step_valid_save_path = 'common-datasets/preprocessed_datasets/second_step_valid.json'

# 第二次推断的输出数据保存位置
second_step_inference_train_save_path = 'inference_results/mrc_results/in_use/second_step/train_results.json'
second_step_inference_valid_save_path = 'inference_results/mrc_results/in_use/second_step/valid_results.json'

if __name__ == '__main__':
    # binary_cls_data_processor_main()
    # mrc_data_processor_main()
    # mrc_predict_main(first_step_valid_save_path, first_step_inference_valid_save_path)
    pass

