# -*- coding: utf - 8 -*-

import os
import json
import time
import collections
import tensorflow as tf
from absl import logging

from optimization import create_optimizer
from distribu_utils import get_distribution_strategy
from distribu_utils import get_strategy_scope
from model import create_model
from inputs_pipeline import read_and_batch_from_tfrecord
from inputs_pipeline import split_dataset
from inputs_pipeline import save_dataset
from data_processor import inference


class ClassifierTask:

    def __init__(self, kwargs, use_pretrain=None):
        if use_pretrain is None:
            raise ValueError('Param use_pretrain must be pass')

        self.use_pretrain = use_pretrain
        self.distribution_strategy = kwargs['distribution_strategy']
        self.epochs = kwargs['epochs']
        self.max_seq_len = kwargs['max_seq_len']
        self.num_labels = kwargs['num_labels']
        self.total_features = kwargs['total_features']
        self.batch_size = kwargs['batch_size']
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

        self.steps_per_epoch = int(
            (self.total_features * (1 - self.valid_data_ratio)) // self.batch_size
        )
        self.total_train_steps = self.steps_per_epoch * self.epochs
        self.distribution_strategy = get_distribution_strategy(self.distribution_strategy, num_gpus=1)

    def train(self):
        self._ensure_dir_exist(self.model_save_dir)

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
                metrics=['binary_accuracy']
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

        model.fit(
            train_dataset,
            initial_epoch=0,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            verbose=1
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

    def predict(self, predict_dataset):
        with get_strategy_scope(self.distribution_strategy):
            model = create_model(self.num_labels, is_train=False, use_pretrain=False)
            self._load_weights_if_possible(
                model,
                tf.train.latest_checkpoint(self.model_save_dir)
            )

        inference(model, self.inference_result_path)

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
                    log_dir=self.model_save_dir
                )
            )

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


# Global Variables #####
DESC_FILE_PATH = './datasets/desc.json'
MODEL_SAVE_DIR = './saved_models'
TFRECORD_FULL_PATH = './datasets/init_train.tfrecord'
TRAIN_TFRECORD_PATH = './datasets/train.tfrecord'
VALID_TFRECORD_PATH = './datasets/valid.tfrecord'
INFERENCE_RESULTS_DIR = './inference_results'
BATCH_SIZE = 1


def get_model_params():
    # load tfrecord description
    with tf.io.gfile.GFile(DESC_FILE_PATH, mode='r') as reader:
        desc = json.load(reader)
    reader.close()

    return collections.defaultdict(
        lambda: None,
        distribution_strategy='one_device',
        epochs=15,
        max_seq_len=desc['max_seq_len'],
        num_labels=desc['num_labels'],
        total_features=desc['total_features'],
        batch_size=BATCH_SIZE,
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
        valid_tfrecord_path=VALID_TFRECORD_PATH
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    task = ClassifierTask(
        get_model_params(),
        use_pretrain=False
    )
    task.train()
