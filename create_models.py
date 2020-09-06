# -*- coding: utf - 8 -*-

import tensorflow as tf
from transformers import BertConfig, TFBertModel
from custom_losses import squad_loss_fn


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
        pooled_output = tf.keras.layers.Dropout(rate=0.1)(pooled_output)

    # (batch_size, num_labels)
    pred = tf.keras.layers.Dense(num_labels, activation='sigmoid')(pooled_output)

    model = tf.keras.Model(
        inputs=[inputs_ids, inputs_mask, segment_ids, label_ids],
        outputs=pred
    )
    return model


def create_mrc_model(is_train=True, use_pretrain=False):

    # 输入
    inputs_ids = tf.keras.Input((None,), name='inputs_ids', dtype=tf.int64)
    inputs_mask = tf.keras.Input((None,), name='inputs_mask', dtype=tf.int64)
    segment_ids = tf.keras.Input((None,), name='segment_ids', dtype=tf.int64)

    # 用于计算 loss

    if use_pretrain:
        # TODO: 使用全局变量或局部变量替换掉这里固定的字符串
        bert_model = TFBertModel.from_pretrained('bert-base-chinese')
    else:
        # 不加载预训练模型，一般在本机测试使用
        # TODO: 使用全局变量或局部变量替换掉这里固定的字符串
        bert_config = BertConfig.from_json_file('./config/bert-base-chinese-config.json')
        bert_model = TFBertModel(bert_config)

    bert_output = bert_model([inputs_ids, inputs_mask, segment_ids], training=is_train)

    # (batch_size, seq_len, hidden_size)
    embedding = bert_output[0]

    # (batch_size, seq_len, 1)
    start_logits = tf.keras.layers.Dense(1, use_bias=False)(embedding)
    # (batch_size, seq_len)
    start_logits = tf.keras.layers.Flatten(name='start_logits')(start_logits)

    end_logits = tf.keras.layers.Dense(1, use_bias=False)(embedding)
    end_logits = tf.keras.layers.Flatten(name='end_logits')(end_logits)

    model = tf.keras.Model(
        inputs=[inputs_ids, inputs_mask, segment_ids],
        outputs=[start_logits, end_logits]
    )

    return model
