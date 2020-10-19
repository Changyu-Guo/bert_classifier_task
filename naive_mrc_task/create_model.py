# -*- coding: utf - 8 -*-

import tensorflow as tf
from transformers import BertConfig, TFBertModel


def create_model(is_train=True, use_net_pretrain=False):

    # 输入
    inputs_ids = tf.keras.Input((None,), name='inputs_ids', dtype=tf.int64)
    inputs_mask = tf.keras.Input((None,), name='inputs_mask', dtype=tf.int64)
    segment_ids = tf.keras.Input((None,), name='segment_ids', dtype=tf.int64)

    # 从网络上加载模型
    if use_net_pretrain:
        core_bert = TFBertModel.from_pretrained('bert-base-chinese')

    # 加载本地模型
    else:
        bert_config = BertConfig.from_json_file('../bert-base-chinese/bert_config.json')
        core_bert = TFBertModel(bert_config)
        checkpoint = tf.train.Checkpoint(model=core_bert)
        checkpoint.restore('../bert-base-chinese/bert_model.ckpt').assert_consumed()

    bert_output = core_bert(
        inputs={
            'input_ids': inputs_ids,
            'attention_mask': inputs_mask,
            'token_type_ids': segment_ids
        },
        training=is_train,
        return_dict=True
    )

    # 最后一层所有输出
    # (batch_size, seq_len, hidden_size)
    embeddings = bert_output['last_hidden_state']

    # (batch_size, seq_len, 1)
    start_logits = tf.keras.layers.Dense(1, use_bias=False)(embeddings)
    # (batch_size, seq_len)
    start_logits = tf.keras.layers.Flatten(name='start_logits')(start_logits)

    # (batch_size, seq_len, 1)
    end_logits = tf.keras.layers.Dense(1, use_bias=False)(embeddings)
    # (batch_size, seq_len)
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
