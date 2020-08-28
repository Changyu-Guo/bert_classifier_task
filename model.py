# -*- coding: utf - 8 -*-

import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertConfig
from optimization import create_optimizer
from absl import logging
from utils import get_distribution_strategy, get_strategy_scope
from inputs_pipeline import read_and_batch_from_tfrecord, load_desc

logging.set_verbosity(logging.INFO)

desc_save_path = './datasets/desc.json'


def create_model(num_labels, is_train):
    inputs_ids = tf.keras.Input((None,), name='inputs_ids', dtype=tf.int64)
    inputs_mask = tf.keras.Input((None,), name='inputs_mask', dtype=tf.int64)
    segment_ids = tf.keras.Input((None,), name='segment_ids', dtype=tf.int64)
    label_ids = tf.keras.Input([num_labels], name='label_ids', dtype=tf.int64)

    # bert_config = BertConfig.from_json_file('./bert-base-chinese-config.json')
    # un_pretrained_bert_model = TFBertModel(bert_config)
    pretrained_bert_model = TFBertModel.from_pretrained('bert-base-chinese')

    # bert_output = un_pretrained_bert_model([inputs_ids, inputs_mask, segment_ids], training=is_train)
    bert_output = pretrained_bert_model([inputs_ids, inputs_mask, segment_ids], training=is_train)

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


def train(
        distribution_strategy,
        num_labels,
        max_seq_len,
        epochs,
        batch_size,
        total_features,
        model_dir,
        data_path
):
    if not tf.io.gfile.exists(model_dir):
        tf.io.gfile.makedirs(model_dir)

    steps_per_epoch = int(total_features // batch_size)
    num_train_steps = steps_per_epoch * epochs

    distribution_strategy = get_distribution_strategy(distribution_strategy, num_gpus=1)
    with get_strategy_scope(distribution_strategy):
        model = create_model(num_labels, is_train=True)
        optimizer = _create_optimizer(num_train_steps)
        checkpoint = tf.train.Checkpoint(
            model=model,
            optimizer=optimizer
        )
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            logging.info('Load checkpoint {} from {}'.format(latest_checkpoint, model_dir))

        model.compile(
            optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['binary_accuracy']
        )

    train_dataset = read_and_batch_from_tfrecord(
        data_path,
        max_seq_len,
        num_labels,
        shuffle=True,
        repeat=True,
        batch_size=batch_size
    )

    model.fit(
        x=train_dataset,
        initial_epoch=0,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )

    checkpoint.save('./saved_models/cls_bert_checkpoint')


def predict(
        distribution_strategy,
        num_labels,
        max_seq_len,
        epochs,
        batch_size,
        total_features,
        model_dir,
        data_path
):

    steps_per_epoch = int(total_features // batch_size)
    num_train_steps = steps_per_epoch * epochs

    distribution_strategy = get_distribution_strategy(distribution_strategy, num_gpus=1)
    with get_strategy_scope(distribution_strategy):
        model = create_model(num_labels, is_train=True)
        optimizer = _create_optimizer(num_train_steps)
        checkpoint = tf.train.Checkpoint(
            model=model,
            optimizer=optimizer
        )
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            logging.info('Load checkpoint {} from {}'.format(latest_checkpoint, model_dir))

        model.compile(optimizer)

    train_dataset = read_and_batch_from_tfrecord(
        data_path,
        max_seq_len,
        num_labels,
        shuffle=True,
        repeat=True,
        batch_size=batch_size
    )

    for data in train_dataset:
        ret = model.predict(data)
        ret = tf.where(ret > 0.5, 1, 0)
        print(ret)

        break


def _create_optimizer(num_train_steps):
    optimizer = create_optimizer(
        init_lr=1e-4,
        num_train_steps=num_train_steps,
        num_warmup_steps=int(num_train_steps // 10),
        end_lr=0.0,
        optimizer_type='adamw'
    )
    return optimizer


def get_params():
    desc = load_desc(desc_save_path)
    num_labels = desc['num_labels']
    max_seq_len = desc['max_seq_len']
    total_features = desc['total_features']
    return dict(
        distribution_strategy='one_device',
        epochs=15,
        num_labels=num_labels,
        max_seq_len=max_seq_len,
        total_features=total_features,
        batch_size=32,
        model_dir='./saved_models',
        data_path='./datasets/init_train.tfrecord'
    )


if __name__ == '__main__':
    train(**get_params())
