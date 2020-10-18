# -*- coding: utf - 8 -*-

import tensorflow as tf
from transformers import BertConfig, TFBertModel


def create_model(is_train=True, use_pretrain=False):
    inputs_ids = tf.keras.Input((None,), name='inputs_ids', dtype=tf.int64)
    inputs_mask = tf.keras.Input((None,), name='inputs_mask', dtype=tf.int64)
    segment_ids = tf.keras.Input((None,), name='segment_ids', dtype=tf.int64)

    if use_pretrain:
        bert_model = TFBertModel.from_pretrained('bert-base-chinese')
    else:
        bert_config = BertConfig.from_json_file('../bert-base-chinese/bert_config.json')
        bert_model = TFBertModel(bert_config)
        checkpoint = tf.train.Checkpoint(model=bert_model)
        checkpoint.restore('../bert-base-chinese/bert_model.ckpt').assert_consumed()

    bert_output = bert_model(
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
