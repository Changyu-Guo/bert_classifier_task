# -*- coding: utf - 8 -*-

import tensorflow as tf
from transformers import TFBertModel, BertConfig
from optimization import create_optimizer
from absl import logging
from utils import get_distribution_strategy, get_strategy_scope
from inputs_pipeline import read_and_batch_from_tfrecord

logging.set_verbosity(logging.INFO)


def create_model(num_labels, is_train):
    inputs_ids = tf.keras.Input((None,), name='inputs_ids', dtype=tf.int64)
    inputs_mask = tf.keras.Input((None,), name='inputs_mask', dtype=tf.int64)
    segment_ids = tf.keras.Input((None,), name='segment_ids', dtype=tf.int64)
    label_ids = tf.keras.Input((None,), batch_size=None, name='label_ids', dtype=tf.int64)

    bert_config = BertConfig.from_json_file('./bert-base-chinese-config.json')
    # pretrained_bert_model = TFBertModel.from_pretrained('bert-base-chinese')
    pretrained_bert_model = TFBertModel(bert_config)

    bert_output = pretrained_bert_model([inputs_ids, inputs_mask, segment_ids])

    pooled_output = bert_output[1]
    if is_train:
        pooled_output = tf.nn.dropout(pooled_output, rate=0.1)

    labels_dense = tf.keras.layers.Dense(num_labels, activation='sigmoid')
    logits = labels_dense(pooled_output)

    model = tf.keras.Model(inputs={
        'inputs_ids': inputs_ids,
        'inputs_mask': inputs_mask,
        'segment_ids': segment_ids,
        'label_ids': label_ids
    }, outputs=logits)
    loss = tf.keras.losses.binary_crossentropy(label_ids, logits, from_logits=False)
    model.add_loss(loss)
    # metric = tf.keras.metrics.binary_accuracy(tf.transpose(label_ids), logits, threshold=0.5)
    # model.add_metric(metric)
    return model


def train(
        distribution_strategy,
        num_labels,
        max_seq_len,
        num_train_steps,
        batch_size,
        model_dir,
        data_path
):
    if not tf.io.gfile.exists(model_dir):
        tf.io.gfile.makedirs(model_dir)
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
        batch_size=None
    )

    callbacks = _create_callback()

    model.fit(
        x=train_dataset,
        y=None,
        batch_size=batch_size,
        initial_epoch=0,
        steps_per_epoch=num_train_steps,
        callbacks=callbacks,
        verbose=1
    )

    checkpoint.save('./saved_models/cls_bert_checkpoint')


def _create_optimizer(num_train_steps):
    optimizer = create_optimizer(
        init_lr=1e-4,
        num_train_steps=num_train_steps,
        num_warmup_steps=int(num_train_steps // 10),
        end_lr=0.0,
        optimizer_type='adamw'
    )
    return optimizer


def _create_callback():
    return []


def get_params():
    return dict(
        distribution_strategy='one_device',
        num_train_steps=20,
        max_seq_len=128,
        num_labels=53,
        batch_size=2,
        model_dir='./saved_models',
        data_path='./datasets/init_train.tfrecord'
    )


if __name__ == '__main__':
    train(**get_params())
