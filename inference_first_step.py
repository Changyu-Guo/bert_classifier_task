# -*- coding: utf - 8 -*-

import json
import collections
import tensorflow as tf
import tokenization
from utils.distribu_utils import get_distribution_strategy
from utils.distribu_utils import get_strategy_scope
from create_models import create_mrc_model
from data_processors.squad_processor_for_inference import FeatureWriter
from data_processors.squad_processor_for_inference import read_squad_examples
from data_processors.squad_processor_for_inference import convert_examples_to_features
from data_processors.squad_processor_for_inference import postprocess_output
from data_processors.inputs_pipeline import read_and_batch_from_squad_tfrecord
from data_processors.inputs_pipeline import map_data_to_mrc_predict_task

MAX_SEQ_LEN = 165
MAX_QUERY_LEN = 45
DOC_STRIDE = 128
PREDICT_BATCH_SIZE = 128
N_BEST_SIZE = 20
MAX_ANSWER_LENGTH = 30
INFERENCE_MODEL_DIR = 'saved_models/mrc_models/mrc_v2_epochs_3'
VOCAB_FILE_PATH = 'vocabs/bert-base-chinese-vocab.txt'

# first step json file
first_step_train_save_path = 'datasets/preprocessed_datasets/first_step_train.json'
first_step_valid_save_path = 'datasets/preprocessed_datasets/first_step_valid.json'


first_step_train_tfrecord_save_path = 'datasets/tfrecord_datasets/first_step_train.tfrecord'
first_step_valid_tfrecord_save_path = 'datasets/tfrecord_datasets/first_step_valid.tfrecord'

# first step inference results
first_step_inference_train_save_path = 'inference_results/mrc_results/in_use/first_step/train_results.json'
first_step_inference_valid_save_path = 'inference_results/mrc_results/in_use/first_step/valid_results.json'


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

# with tf.io.gfile.GFile(first_step_valid_save_path, mode='r') as reader:
#     input_data = json.load(reader)['data']
# reader.close()
#
#
# examples = read_squad_examples(
#     input_data
# )
#
# writer = FeatureWriter(
#     filename=first_step_valid_tfrecord_save_path,
#     is_training=False
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
#     filename=first_step_valid_tfrecord_save_path,
#     max_seq_len=MAX_SEQ_LEN,
#     is_training=False,
#     repeat=False,
#     batch_size=128
# )
#
# all_results = []
# for data in dataset:
#     unique_ids = data.pop('unique_ids')
#     model_output = model.predict(
#         map_data_to_mrc_predict_task(data)
#     )
#
#     start_logits = model_output['start_logits']
#     end_logits = model_output['end_logits']
#
#     for result in get_raw_results(dict(
#         unique_ids=unique_ids,
#         start_logits=start_logits,
#         end_logits=end_logits
#     )):
#         all_results.append(result)
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
#     input_data[0]['paragraphs'][i]['pred_relations'] = []
#
# for predict in all_predictions:
#     context_index = predict['context_index']
#     pred_answer = predict['pred_answer']
#     relation = predict['relation']
#
#     input_data[0]['paragraphs'][context_index]['pred_relations'].append(
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
# with tf.io.gfile.GFile(first_step_inference_valid_save_path, mode='w') as writer:
#     writer.write(json.dumps(input_data, ensure_ascii=False, indent=2))
# writer.close()

with tf.io.gfile.GFile(first_step_train_save_path, mode='r') as reader:
    input_data = json.load(reader)['data']
reader.close()


examples = read_squad_examples(
    input_data
)

writer = FeatureWriter(
    filename=first_step_train_tfrecord_save_path,
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
    filename=first_step_train_tfrecord_save_path,
    max_seq_len=MAX_SEQ_LEN,
    is_training=False,
    repeat=False,
    batch_size=128
)

all_results = []
for data in dataset:
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
    input_data[0]['paragraphs'][i]['pred_relations'] = []

for predict in all_predictions:
    context_index = predict['context_index']
    pred_answer = predict['pred_answer']
    relation = predict['relation']

    input_data[0]['paragraphs'][context_index]['pred_relations'].append(
        {
            'relation': relation,
            'subject': pred_answer
        }
    )

input_data = {
    'data': input_data
}

with tf.io.gfile.GFile(first_step_inference_train_save_path, mode='w') as writer:
    writer.write(json.dumps(input_data, ensure_ascii=False, indent=2))
writer.close()