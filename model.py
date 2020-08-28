# -*- coding: utf - 8 -*-

import tensorflow as tf
from transformers import TFBertModel
from transformers import create_optimizer
from absl import logging
from utils import get_distribution_strategy, get_strategy_scope
from inputs_pipeline import read_and_batch_from_tfrecord

logging.set_verbosity(logging.INFO)


def create_model(num_labels, is_train):
    inputs_ids = tf.keras.Input((None,), name='inputs_ids', dtype=tf.int64)
    inputs_mask = tf.keras.Input((None,), name='inputs_mask', dtype=tf.int64)
    segment_ids = tf.keras.Input((None,), name='segment_ids', dtype=tf.int64)
    label_ids = tf.keras.Input((None,), name='label_ids', dtype=tf.int64)

    pretrained_bert_model = TFBertModel.from_pretrained('bert-base-chinese')

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
    })
    loss = tf.nn.sigmoid_cross_entropy_with_logits(label_ids, logits)
    model.add_loss(loss)
    model.add_metric('accuracy')
    return model


def train(
        distribution_strategy='mirror',
        num_labels=53,
        max_seq_len=128,
        num_train_steps=100000,
        steps_between_eval=10000,
        model_dir='',
        data_path=''
):
    distribution_strategy = get_distribution_strategy(distribution_strategy, num_gpus=1)
    with get_strategy_scope(distribution_strategy):
        model = create_model(num_labels, is_train=True)
        optimizer = _create_optimizer(num_train_steps)

        cur_step = 0

        checkpoint = tf.train.Checkpoint(
            model=model,
            optimizer=optimizer
        )
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            logging.info('Load checkpoint {} from {}'.format(latest_checkpoint, model_dir))
            cur_step = optimizer.iterations.numpy()

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

    while cur_step < num_train_steps:
        remaining_steps = num_train_steps - cur_step
        train_steps_per_eval = remaining_steps if remaining_steps < steps_between_eval else steps_between_eval
        cur_iteration = cur_step // steps_between_eval

        his = model.fit(
            train_dataset,
            initial_epoch=cur_iteration,
            steps_per_epoch=train_steps_per_eval,
            callbacks=callbacks,
            verbose=1
        )

        cur_step += train_steps_per_eval


def _create_optimizer(num_train_steps):
    optimizer, _ = create_optimizer(
        init_lr=1e-4,
        num_train_steps=num_train_steps,
        num_warmup_steps=int(num_train_steps // 10),
        min_lr_ratio=0.0,
        adam_epsilon=1e-6,
        weight_decay_rate=0.1,
        include_in_weight_decay=None
    )
    return optimizer


def _create_callback():
    return []


def params():
    return dict(
        distribution_strategy='mirrored',
        num_train_steps=1000,
        steps_between_eval=100,
        max_seq_len=128,
        num_labels=53,
        model_dir='./saved_models',
        data_path='./datasets/init_train.tfrecord'
    )


if __name__ == '__main__':
    train(params())
