# -*- coding: utf - 8 -*-

import collections
import json

import tensorflow as tf

import tokenization
from create_models import create_mrc_model
from data_processors.inputs_pipeline import map_data_to_mrc_predict_task
from data_processors.inputs_pipeline import read_and_batch_from_squad_tfrecord
from data_processors.squad_processor_for_first_inference import FeatureWriter
from data_processors.squad_processor_for_first_inference import convert_examples_to_features
from data_processors.squad_processor_for_first_inference import postprocess_output
from data_processors.squad_processor_for_first_inference import read_squad_examples
from utils.distribu_utils import get_distribution_strategy
from utils.distribu_utils import get_strategy_scope

MAX_SEQ_LEN = 165
MAX_QUERY_LEN = 45
DOC_STRIDE = 128
PREDICT_BATCH_SIZE = 128
N_BEST_SIZE = 20
MAX_ANSWER_LENGTH = 30

# MRC Model
INFERENCE_MODEL_DIR = 'saved_models/mrc_models/mrc_v2_epochs_3'

# tokenizer vocab
VOCAB_FILE_PATH = 'vocabs/bert-base-chinese-vocab.txt'

# 推断的输入
train_data_before_first_step_save_path = \
    'common-datasets/preprocessed_datasets/before_mrc_first_step/in_use/first_step_' \
    'train.json'
valid_data_before_first_step_save_path = 'common-datasets/preprocessed_datasets/before_mrc_first_step/in_use/first_step_' \
                                         'valid.json'

train_tfrecord_before_first_step_save_path = 'common-datasets/tfrecord_datasets/first_step_train.tfrecord'
valid_tfrecord_before_first_step_save_path = 'common-datasets/tfrecord_datasets/first_step_valid.tfrecord'

# first step inference inference_results
train_data_after_first_step_save_path = 'inference_results/mrc_results/in_use/first_step/train_results.json'
valid_data_after_first_step_save_path = 'inference_results/mrc_results/in_use/first_step/valid_results.json'


def get_raw_results(predictions):
    RawResult = collections.namedtuple(
        'RawResult',
        ['unique_id', 'start_logits', 'end_logits']
    )
    for unique_id, start_logits, end_logits in zip(
            predictions['unique_ids'],
            predictions['start_logits'],
            predictions['end_logits']
    ):
        yield RawResult(
            unique_id=unique_id.numpy(),
            start_logits=start_logits.tolist(),
            end_logits=end_logits.tolist()
        )


# 加载模型
distribution_strategy = get_distribution_strategy('one_device')

with get_strategy_scope(distribution_strategy):
    model = create_mrc_model(
        max_seq_len=MAX_SEQ_LEN,
        is_train=False, use_pretrain=False
    )
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(
        tf.train.latest_checkpoint(
            INFERENCE_MODEL_DIR
        )
    )

tokenizer = tokenization.FullTokenizer(
    vocab_file=VOCAB_FILE_PATH, do_lower_case=True
)

# 在使用 MRC 推断之前，数据必须提前处理为 SQuAD 类型
# with tf.io.gfile.GFile(valid_data_before_first_step_save_path, mode='r') as reader:
#     input_data = json.load(reader)['data']
# reader.close()
#
# # 读取所有 QA Example
# examples = read_squad_examples(
#     input_data
# )
#
# print(len(examples))
#
# # 转为 TFRecord
# writer = FeatureWriter(
#     filename=valid_tfrecord_before_first_step_save_path,
#     is_training=False  # 推断过程中不知道答案，所以设置为 False
# )
#
# all_features = []
#
#
# def _append_feature(feature, is_padding):
#     if not is_padding:
#         all_features.append(feature)
#     writer.process_feature(feature)
#
#
# convert_examples_to_features(
#     examples, tokenizer, max_seq_length=MAX_SEQ_LEN,
#     doc_stride=DOC_STRIDE, max_query_length=MAX_QUERY_LEN,
#     is_training=False, output_fn=_append_feature, batch_size=128
# )
# writer.close()
#
# dataset = read_and_batch_from_squad_tfrecord(
#     filename=valid_tfrecord_before_first_step_save_path,
#     max_seq_len=MAX_SEQ_LEN,
#     is_training=False,
#     repeat=False,
#     batch_size=128
# )
#
# all_results = []
# for index, data in enumerate(dataset):
#     unique_ids = data.pop('unique_ids')
#     model_output = model.predict(
#         map_data_to_mrc_predict_task(data)
#     )
#
#     start_logits = model_output['start_logits']
#     end_logits = model_output['end_logits']
#
#     for result in get_raw_results(dict(
#             unique_ids=unique_ids,
#             start_logits=start_logits,
#             end_logits=end_logits
#     )):
#         all_results.append(result)
#
#     print(index)
#
# all_predictions = postprocess_output(
#     input_data,
#     examples,
#     all_features,
#     all_results,
#     n_best_size=N_BEST_SIZE,
#     max_answer_length=MAX_ANSWER_LENGTH,
#     do_lower_case=True,
#     verbose=False
# )
#
# for i in range(len(input_data[0]['paragraphs'])):
#     input_data[0]['paragraphs'][i]['pred_sros'] = []
#
# for predict in all_predictions:
#     context_index = predict['context_index']
#     pred_answer = predict['pred_answer']
#     relation = predict['relation']
#
#     input_data[0]['paragraphs'][context_index]['pred_sros'].append(
#         {
#             'relation': relation,
#             'subject': pred_answer
#         }
#     )
#
# input_data = {
#     'data': input_data
# }
#
# with tf.io.gfile.GFile(valid_data_after_first_step_save_path, mode='w') as writer:
#     writer.write(json.dumps(input_data, ensure_ascii=False, indent=2))
# writer.close()

with tf.io.gfile.GFile(train_data_before_first_step_save_path, mode='r') as reader:
    input_data = json.load(reader)['data']
reader.close()


examples = read_squad_examples(
    input_data
)
print(len(examples))

writer = FeatureWriter(
    filename=train_tfrecord_before_first_step_save_path,
    is_training=False
)

all_features = []


def _append_feature(feature, is_padding):
    if not is_padding:
        all_features.append(feature)
    writer.process_feature(feature)


convert_examples_to_features(
    examples, tokenizer, max_seq_length=MAX_SEQ_LEN,
    doc_stride=DOC_STRIDE, max_query_length=MAX_QUERY_LEN,
    is_training=False, output_fn=_append_feature, batch_size=128
)
writer.close()

dataset = read_and_batch_from_squad_tfrecord(
    filename=train_tfrecord_before_first_step_save_path,
    max_seq_len=MAX_SEQ_LEN,
    is_training=False,
    repeat=False,
    batch_size=128
)

all_results = []
for index, data in enumerate(dataset):
    unique_ids = data.pop('unique_ids')
    model_output = model.predict(
        map_data_to_mrc_predict_task(data)
    )

    start_logits = model_output['start_logits']
    end_logits = model_output['end_logits']

    for result in get_raw_results(dict(
        unique_ids=unique_ids,
        start_logits=start_logits,
        end_logits=end_logits
    )):
        all_results.append(result)

    print(index)

all_predictions = postprocess_output(
    input_data,
    examples,
    all_features,
    all_results,
    n_best_size=N_BEST_SIZE,
    max_answer_length=MAX_ANSWER_LENGTH,
    do_lower_case=True,
    verbose=False
)

for i in range(len(input_data[0]['paragraphs'])):
    input_data[0]['paragraphs'][i]['pred_sros'] = []

for predict in all_predictions:
    context_index = predict['context_index']
    pred_answer = predict['pred_answer']
    relation = predict['relation']

    input_data[0]['paragraphs'][context_index]['pred_sros'].append(
        {
            'relation': relation,
            'subject': pred_answer
        }
    )

input_data = {
    'data': input_data
}

with tf.io.gfile.GFile(train_data_after_first_step_save_path, mode='w') as writer:
    writer.write(json.dumps(input_data, ensure_ascii=False, indent=2))
writer.close()
