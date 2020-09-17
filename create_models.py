# -*- coding: utf - 8 -*-

import tensorflow as tf
from transformers import BertConfig, TFBertModel


def create_multi_label_cls_model(num_labels, is_train=True, use_pretrain=False):
    inputs_ids = tf.keras.Input((None,), name='inputs_ids', dtype=tf.int64)
    inputs_mask = tf.keras.Input((None,), name='inputs_mask', dtype=tf.int64)
    segment_ids = tf.keras.Input((None,), name='segment_ids', dtype=tf.int64)

    if use_pretrain:
        bert_model = TFBertModel.from_pretrained('bert-base-chinese')
    else:
        bert_config = BertConfig.from_json_file('config/bert-base-chinese-config.json')
        bert_model = TFBertModel(bert_config)

    bert_output = bert_model({
        'input_ids': inputs_ids,
        'attention_mask': inputs_mask,
        'token_type_ids': segment_ids
    }, training=is_train)

    # (batch_size, hidden_size)
    pooled_output = bert_output[1]
    if is_train:
        pooled_output = tf.keras.layers.Dropout(rate=0.1)(pooled_output)

    # (batch_size, num_labels)
    probs = tf.keras.layers.Dense(num_labels, activation='sigmoid')(pooled_output)

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


def create_mrc_model(max_seq_len, is_train=True, use_pretrain=False):
    bert_config_file_path = 'config/bert-base-chinese-config.json'

    # 输入
    inputs_ids = tf.keras.Input((max_seq_len,), name='inputs_ids', dtype=tf.int64)
    inputs_mask = tf.keras.Input((max_seq_len,), name='inputs_mask', dtype=tf.int64)
    segment_ids = tf.keras.Input((max_seq_len,), name='segment_ids', dtype=tf.int64)

    if use_pretrain:
        # TODO: 使用全局变量或局部变量替换掉这里固定的字符串
        bert_model = TFBertModel.from_pretrained('bert-base-chinese')
    else:
        # 不加载预训练模型，一般在本机测试或者是推断的时候使用
        # TODO: 使用全局变量或局部变量替换掉这里固定的字符串
        bert_config = BertConfig.from_json_file(bert_config_file_path)
        bert_model = TFBertModel(bert_config)

    bert_output = bert_model({
        'input_ids': inputs_ids,
        'attention_mask': inputs_mask,
        'token_type_ids': segment_ids
    }, training=is_train)

    # 最后一层所有输出
    # (batch_size, seq_len, hidden_size)
    embedding = bert_output[0]

    start_logits = tf.keras.layers.Dense(1, use_bias=False)(embedding)
    start_logits = tf.keras.layers.Flatten(name='start_logits')(start_logits)

    end_logits = tf.keras.layers.Dense(1, use_bias=False)(embedding)
    end_logits = tf.keras.layers.Flatten(name='end_logits')(end_logits)

    model = tf.keras.Model(
        inputs={
            'inputs_ids': inputs_ids,
            'inputs_mask': inputs_mask,
            'segment_ids': segment_ids
        },
        outputs={
            'start_logits': start_logits,
            'end_logits': end_logits
        }
    )

    return model


def create_binary_cls_model(is_train=True, use_pretrain=False):
    inputs_ids = tf.keras.Input((None,), name='inputs_ids', dtype=tf.int64)
    inputs_mask = tf.keras.Input((None,), name='inputs_mask', dtype=tf.int64)
    segment_ids = tf.keras.Input((None,), name='segment_ids', dtype=tf.int64)

    if use_pretrain:
        bert_model = TFBertModel.from_pretrained('bert-base-chinese')
    else:
        bert_config = BertConfig.from_json_file('config/bert-base-chinese-config.json')
        bert_model = TFBertModel(bert_config)
    bert_output = bert_model({
        'input_ids': inputs_ids,
        'attention_mask': inputs_mask,
        'token_type_ids': segment_ids
    }, training=is_train)
    pooled_output = bert_output[1]
    if is_train:
        pooled_output = tf.keras.layers.Dropout(rate=0.1)(pooled_output)
    prob = tf.keras.layers.Dense(1, activation='sigmoid')(pooled_output)
    model = tf.keras.Model(
        inputs={
            'inputs_ids': inputs_ids,
            'inputs_mask': inputs_mask,
            'segment_ids': segment_ids
        },
        outputs=prob
    )

    return model


if __name__ == '__main__':
    """
        该文件已经过审查
    """
    pass
