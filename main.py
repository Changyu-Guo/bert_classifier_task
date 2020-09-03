# -*- coding: utf - 8 -*-

import os
import json
import time
import collections
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from absl import logging

from optimization import create_optimizer
from distribu_utils import get_distribution_strategy
from distribu_utils import get_strategy_scope
from multi_label_cls_model import create_model
from inputs_pipeline import read_and_batch_from_tfrecord
from inputs_pipeline import split_dataset
from inputs_pipeline import save_dataset
from data_processor import (
    inference,
    inference_tfrecord,
    calculate_tfrecord_prf,
    log_inference_tfrecord_time
)


class ClassifierTask:

    def __init__(self, kwargs, use_pretrain=None, batch_size=None, inference_type=None):
        if use_pretrain is None or batch_size is None:
            raise ValueError('Param use_pretrain and batch_size must be pass')
        self.batch_size = batch_size
        self.use_pretrain = use_pretrain
        self.inference_type = inference_type

        self.distribution_strategy = kwargs['distribution_strategy']
        self.epochs = kwargs['epochs']
        self.max_seq_len = kwargs['max_seq_len']
        self.num_labels = kwargs['num_labels']
        self.total_features = kwargs['total_features']
        self.model_save_dir = kwargs['model_save_dir']
        self.tfrecord_path = kwargs['tfrecord_path']
        self.train_tfrecord_path = kwargs['train_tfrecord_path']
        self.valid_tfrecord_path = kwargs['valid_tfrecord_path']
        self.enable_checkpointing = kwargs['enable_checkpointing']
        self.enable_tensorboard = kwargs['enable_tensorboard']
        self.init_lr = kwargs['init_lr']
        self.end_lr = kwargs['end_lr']
        self.warmup_steps_ratio = kwargs['warmup_steps_ratio']
        self.valid_data_ratio = kwargs['valid_data_ratio']
        self.inference_result_path = kwargs['inference_result_path']
        self.tensorboard_log_dir = kwargs['tensorboard_log_dir']
        self.history_save_path = kwargs['history_save_path']

        self.steps_per_epoch = int(
            (self.total_features * (1 - self.valid_data_ratio)) // self.batch_size
        )
        self.total_train_steps = self.steps_per_epoch * self.epochs
        self.distribution_strategy = get_distribution_strategy(self.distribution_strategy, num_gpus=1)

    def train(self):
        self._ensure_dir_exist(self.model_save_dir)
        self._ensure_dir_exist(self.tensorboard_log_dir)

        # 在 distribution strategy scope 下定义:
        #   1. model
        #   2. optimizer
        #   3. load checkpoint
        #   4. compile
        with get_strategy_scope(self.distribution_strategy):

            model = create_model(self.num_labels, is_train=True, use_pretrain=self.use_pretrain)
            optimizer = self._create_optimizer()

            # load checkpoint
            checkpoint = tf.train.Checkpoint(
                model=model,
                optimizer=optimizer
            )
            latest_checkpoint = tf.train.latest_checkpoint(self.model_save_dir)
            if latest_checkpoint:
                checkpoint.restore(latest_checkpoint)
                logging.info('Load checkpoint {} from {}'.format(latest_checkpoint, self.model_save_dir))

            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=[
                    tfa.metrics.MultiLabelConfusionMatrix(
                        num_classes=self.num_labels
                    )
                ]
            )

        if tf.io.gfile.exists(self.train_tfrecord_path) and \
                tf.io.gfile.exists(self.valid_tfrecord_path):
            train_dataset = read_and_batch_from_tfrecord(
                self.train_tfrecord_path,
                max_seq_len=self.max_seq_len,
                num_labels=self.num_labels,
                shuffle=True,
                repeat=True,
                batch_size=self.batch_size
            )
            valid_dataset = read_and_batch_from_tfrecord(
                self.valid_tfrecord_path,
                max_seq_len=self.max_seq_len,
                num_labels=self.num_labels,
                shuffle=False,
                repeat=False,
                batch_size=self.batch_size
            )

        # 没有切分后的数据
        # 读取原始数据并切分
        else:
            # load dataset
            dataset = read_and_batch_from_tfrecord(
                filename=self.tfrecord_path,
                max_seq_len=self.max_seq_len,
                num_labels=self.num_labels,
                shuffle=True,
                repeat=False,
                batch_size=None
            )

            # 切分数据集
            train_dataset, valid_dataset = split_dataset(
                dataset,
                valid_ratio=self.valid_data_ratio,
                total_features=self.total_features
            )

            save_dataset(train_dataset, self.train_tfrecord_path)
            save_dataset(valid_dataset, self.valid_tfrecord_path)

            train_dataset = train_dataset.repeat().batch(self.batch_size)
            valid_dataset = valid_dataset.batch(self.batch_size)

        callbacks = self._create_callbacks()

        his = model.fit(
            train_dataset,
            initial_epoch=0,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
            validation_data=valid_dataset
        )

        checkpoint.save(os.path.join(self.model_save_dir, 'train_end_checkpoint'))

    def eval(self, dataset):
        with get_strategy_scope(self.distribution_strategy):
            model = create_model(self.num_labels, is_train=False, use_pretrain=False)
            self._load_weights_if_possible(
                model,
                tf.train.latest_checkpoint(self.model_save_dir)
            )

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(threshold=0.5)
            ]
        )
        model.evaluate(dataset)

    def predict(self):
        with get_strategy_scope(self.distribution_strategy):
            model = create_model(self.num_labels, is_train=False, use_pretrain=False)
            self._load_weights_if_possible(
                model,
                tf.train.latest_checkpoint(self.model_save_dir)
            )

        inference(model, self.inference_result_path)

    def predict_with_tfrecord(self):

        # restore model
        with get_strategy_scope(self.distribution_strategy):
            model = create_model(self.num_labels, is_train=False, use_pretrain=False)
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(tf.train.latest_checkpoint('./saved_models/classifier_epoch_30'))

        dataset = read_and_batch_from_tfrecord(
            filename=self.train_tfrecord_path,  # Notice ####
            max_seq_len=self.max_seq_len,
            num_labels=self.num_labels,
            shuffle=False,
            repeat=False,
            batch_size=1  # Notice ####
        )
        for data in dataset:
            model.predict(data)
            break

        # read validation dataset
        print('start...')
        batch_sizes = [2000, 1000, 500, 400, 200, 100, 50, 1]
        for batch_size in batch_sizes:
            dataset = read_and_batch_from_tfrecord(
                filename=self.train_tfrecord_path,  # Notice ####
                max_seq_len=self.max_seq_len,
                num_labels=self.num_labels,
                shuffle=False,
                repeat=False,
                batch_size=batch_size  # Notice ####
            )
            if self.inference_type == 'inference_tfrecord':
                inference_tfrecord(model, self.inference_result_path, dataset)
            elif self.inference_type == 'calculate_tfrecord_prf':
                calculate_tfrecord_prf(model, dataset)
            elif self.inference_type == 'log_inference_tfrecord_time':
                log_inference_tfrecord_time(model, dataset, batch_size)

    def _create_callbacks(self):
        """
            三个重要的回调：
                1. checkpoint (重要)
                2. summary (可选)
                3. earlyStopping (可选)
        """
        callbacks = []
        if self.enable_checkpointing:
            ckpt_path = os.path.join(self.model_save_dir, 'cp-{epoch:04d}.ckpt')
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    ckpt_path, save_weights_only=True, save_best_only=True
                )
            )

        if self.enable_tensorboard:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=self.tensorboard_log_dir
                )
            )

        return callbacks

    def _load_weights_if_possible(self, model, init_weight_path=None):

        if init_weight_path:
            logging.info('Load weights: {}'.format(init_weight_path))
            model.load_weights(init_weight_path)
        else:
            logging.info('Weights not loaded from path: {}'.format(init_weight_path))

    def _create_optimizer(self):
        return create_optimizer(
            init_lr=self.init_lr,
            num_train_steps=self.total_train_steps,
            num_warmup_steps=int(self.total_train_steps * self.warmup_steps_ratio),
            end_lr=self.end_lr,
            optimizer_type='adamw'
        )

    def _ensure_dir_exist(self, _dir):
        if not tf.io.gfile.exists(_dir):
            tf.io.gfile.makedirs(_dir)


class MRCTask:

    def __init__(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def predict(self):
        pass


# Global Variables #####
DESC_FILE_PATH = './datasets/desc.json'
MODEL_SAVE_DIR = './saved_models'
TFRECORD_FULL_PATH = './datasets/init_train.tfrecord'
TRAIN_TFRECORD_PATH = './datasets/train.tfrecord'
VALID_TFRECORD_PATH = './datasets/valid.tfrecord'
INFERENCE_RESULTS_DIR = './inference_results'
TENSORBOARD_LOG_DIR = './logs'
HISTORY_SAVE_PATH = './saved_models/history.csv'


def get_model_params():
    # load tfrecord description
    with tf.io.gfile.GFile(DESC_FILE_PATH, mode='r') as reader:
        desc = json.load(reader)
    reader.close()

    return collections.defaultdict(
        lambda: None,
        distribution_strategy='one_device',
        epochs=30,
        max_seq_len=desc['max_seq_len'],
        num_labels=desc['num_labels'],
        total_features=desc['total_features'],
        model_save_dir=MODEL_SAVE_DIR,
        tfrecord_path='./datasets/init_train.tfrecord',
        init_lr=1e-4,
        end_lr=0.0,
        warmup_steps_ratio=0.1,
        valid_data_ratio=0.1,
        inference_result_path=os.path.join(
            INFERENCE_RESULTS_DIR,
            time.strftime('%Y_%m_%d', time.localtime()) + '_result.txt'),
        train_tfrecord_path=TRAIN_TFRECORD_PATH,
        valid_tfrecord_path=VALID_TFRECORD_PATH,
        enable_tensorboard=True,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        history_save_path=HISTORY_SAVE_PATH
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    task = ClassifierTask(
        get_model_params(),
        use_pretrain=False,  # Notice ###
        batch_size=2,  # Notice ###
        inference_type='log_inference_tfrecord_time'
    )
    task.predict_with_tfrecord()
