# -*- coding: utf - 8 -*-

import tensorflow as tf
from transformers import BertConfig, TFBertModel

inputs_ids = tf.keras.Input((None,), dtype=tf.int64)

config = BertConfig.from_json_file('config/chinese_wwm_roberta_large_config.json')
in_model = TFBertModel(config)

out = in_model(inputs_ids)
out = out[0]

out = tf.keras.layers.Dense(2)(out)

model = tf.keras.Model(inputs=inputs_ids, outputs=out)

checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore('pretrained_model/chinese_roberta_wwm_large_ext_tf/bert_model.ckpt')


print(tf.keras.utils.plot_model(model))
