# -*- coding: utf - 8 -*-

from data_processors import mrc_data_processor
from data_processors import bi_cls_data_processor_s1
from data_processors import commom

if __name__ == '__main__':
    bi_cls_data_processor_s1.postprocess_valid_output(
        all_examples=None,
        batched_origin_is_valid=None,
        batched_pred_is_valid=None,
        results_save_path=None
    )
