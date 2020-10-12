# -*- coding: utf - 8 -*-

import tensorflow as tf
from tensorflow import keras
from transformers import TFBertModel, BertConfig


def create_model(use_net_pretrain, is_train=True):
    inputs_ids = keras.Input((None,), name='inputs_ids', dtype=tf.int64)
    inputs_mask = keras.Input((None,), name='inputs_mask', dtype=tf.int64)
    segment_ids = keras.Input((None,), name='segment_ids', dtype=tf.int64)

    # load bert core
    if use_net_pretrain:
        core_bert = TFBertModel.from_pretrained('bert-base-chinese')
    else:
        bert_config = BertConfig.from_json_file('../bert-base-chinese/bert_config.json')
        core_bert = TFBertModel(bert_config)
        checkpoint = tf.train.Checkpoint(model=core_bert)
        checkpoint.restore(tf.train.latest_checkpoint('../bert-base-chinese')).assert_consumed()

    bert_output = core_bert(
        inputs={
            'input_ids': inputs_ids,
            'attention_mask': inputs_mask,
            'token_type_ids': segment_ids
        },
        training=is_train,
        return_dict=True
    )

    pooled_output = bert_output['pooler_output']
    if is_train:
        pooled_output = tf.keras.layers.Dropout(rate=0.1)(pooled_output)

    probs = tf.keras.layers.Dense(1, activation='sigmoid')(pooled_output)
    model = tf.keras.Model(
        inputs={
            'inputs_ids': inputs_ids,
            'inputs_mask': inputs_mask,
            'segment_ids': segment_ids
        },
        outputs={
            'probs': probs
        }
    )

    return model
