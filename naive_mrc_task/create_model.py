# -*- coding: utf - 8 -*-

import tensorflow as tf
from transformers import BertConfig, TFBertModel


def create_mrc_model(max_seq_len, is_train=True, use_pretrain=False):

    # 输入
    inputs_ids = tf.keras.Input((max_seq_len,), name='inputs_ids', dtype=tf.int64)
    inputs_mask = tf.keras.Input((max_seq_len,), name='inputs_mask', dtype=tf.int64)
    segment_ids = tf.keras.Input((max_seq_len,), name='segment_ids', dtype=tf.int64)

    if use_pretrain:
        core_bert = TFBertModel.from_pretrained('bert-base-chinese')
    else:
        bert_config = BertConfig.from_json_file('../configs/bert-base-chinese-config.json')
        core_bert = TFBertModel(bert_config)

    bert_output = core_bert({
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