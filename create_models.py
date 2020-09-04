# -*- coding: utf - 8 -*-

import tensorflow as tf
from transformers import BertConfig, TFBertModel


def create_cls_model(num_labels, is_train=True, use_pretrain=False):
    inputs_ids = tf.keras.Input((None,), name='inputs_ids', dtype=tf.int64)
    inputs_mask = tf.keras.Input((None,), name='inputs_mask', dtype=tf.int64)
    segment_ids = tf.keras.Input((None,), name='segment_ids', dtype=tf.int64)
    label_ids = tf.keras.Input([num_labels], name='label_ids', dtype=tf.int64)

    if use_pretrain:
        bert_model = TFBertModel.from_pretrained('bert-base-chinese')
    else:
        bert_config = BertConfig.from_json_file('./config/bert-base-chinese-config.json')
        bert_model = TFBertModel(bert_config)

    bert_output = bert_model([inputs_ids, inputs_mask, segment_ids], training=is_train)

    # (batch_size, hidden_size)
    pooled_output = bert_output[1]
    if is_train:
        pooled_output = tf.nn.dropout(pooled_output, rate=0.1)

    # (batch_size, num_labels)
    pred = tf.keras.layers.Dense(num_labels, activation='sigmoid')(pooled_output)

    model = tf.keras.Model(
        inputs=[inputs_ids, inputs_mask, segment_ids, label_ids],
        outputs=pred
    )
    return model


def create_mrc_model(is_train=True, use_pretrain=False):
    inputs_ids = tf.keras.Input((None,), name='inputs_ids', dtype=tf.int64)
    inputs_mask = tf.keras.Input((None,), name='inputs_mask', dtype=tf.int64)
    segment_ids = tf.keras.Input((None,), name='segment_ids', dtype=tf.int64)
    start_position = tf.keras.Input((None,), name='start_position', dtype=tf.int64)
    end_position = tf.keras.Input((None,), name='end_position', dtype=tf.int64)

    if use_pretrain:
        bert_model = TFBertModel.from_pretrained('bert-base-chinese')
    else:
        bert_config = BertConfig.from_json_file('./config/bert-base-chinese-config.json')
        bert_model = TFBertModel(bert_config)

    bert_output = bert_model([inputs_ids, inputs_mask, segment_ids], train=is_train)

    # (batch_size, seq_len, hidden_size)
    pooled_output = bert_output[1]

    if is_train:
        pooled_output = tf.nn.dropout(pooled_output, rate=0.1)

    # (batch_size, seq_len, 2)
    logits = tf.keras.layers.Dense(units=2)

    # (batch_size, seq_len, 1)
    start_logits, end_logits = tf.split(logits, 2, axis=-1)

    # (batch_size, seq_len)
    start_logits = tf.squeeze(start_logits, axis=-1, name='start_logits')
    end_logits = tf.squeeze(end_logits, axis=-1, name='end_logits')

    model = tf.keras.Model(
        inputs=[inputs_ids, inputs_mask, segment_ids, start_position, end_position],
        outputs=[start_logits, end_logits]
    )

    return model

