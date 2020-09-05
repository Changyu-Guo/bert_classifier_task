# -*- coding: utf - 8 -*-

import os
import json
import time
import collections
import tensorflow as tf
from absl import logging
from create_models import create_mrc_model
from distribu_utils import get_distribution_strategy
from distribu_utils import get_strategy_scope
from optimization import create_optimizer
from inputs_pipeline import read_and_batch_from_squad_tfrecord
from inputs_pipeline import split_dataset
from squad_processor import save_squad_dataset


class MRCTask:

    def __init__(self, kwargs, use_pretrain=None, batch_size=None, inference_type=None):

        # param check
        if use_pretrain is None:
            raise ValueError('Param use_pretrain must be passed')

        self.batch_size = batch_size
        self.use_pretrain = use_pretrain
        self.inference_type = inference_type

        self.task_name = kwargs['task_name']
        self.distribution_strategy = kwargs['distribution_strategy']
        self.epochs = kwargs['epochs']
        self.max_seq_len = kwargs['max_seq_len']
        self.total_features = kwargs['total_features']

        self.tfrecord_path = kwargs['tfrecord_path']
        self.train_tfrecord_path = kwargs['train_tfrecord_path']
        self.valid_tfrecord_path = kwargs['valid_tfrecord_path']
        self.valid_data_ratio = kwargs['valid_data_ratio']

        self.enable_checkpointing = kwargs['enable_checkpointing']
        self.enable_tensorboard = kwargs['enable_tensorboard']
        self.init_lr = kwargs['init_lr']
        self.end_lr = kwargs['end_lr']
        self.warmup_steps_ratio = kwargs['warmup_steps_ratio']

        self.time_prefix = kwargs['time_prefix']
        self.model_save_dir = os.path.join(
            kwargs['model_save_dir'],
            self.time_prefix + '_' + self.task_name + '_' + str(self.epochs)
        )
        self.tensorboard_log_dir = os.path.join(
            kwargs['tensorboard_log_dir'],
            self.time_prefix + '_' + self.task_name + '_' + str(self.epochs)
        )

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

            model = create_mrc_model(is_train=True, use_pretrain=self.use_pretrain)
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
                loss=[
                    tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=False,
                    ),
                    tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=False,
                    )
                ]
            )

        if tf.io.gfile.exists(self.train_tfrecord_path) and \
                tf.io.gfile.exists(self.valid_tfrecord_path):
            train_dataset = read_and_batch_from_squad_tfrecord(
                self.train_tfrecord_path,
                max_seq_len=self.max_seq_len,
                shuffle=True,
                repeat=True,
                batch_size=self.batch_size
            )
            valid_dataset = read_and_batch_from_squad_tfrecord(
                self.valid_tfrecord_path,
                max_seq_len=self.max_seq_len,
                shuffle=False,
                repeat=False,
                batch_size=self.batch_size
            )

        # 没有切分后的数据
        # 读取原始数据并切分
        else:
            # load dataset
            dataset = read_and_batch_from_squad_tfrecord(
                filename=self.tfrecord_path,
                max_seq_len=self.max_seq_len,
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

            save_squad_dataset(train_dataset, self.train_tfrecord_path)
            save_squad_dataset(valid_dataset, self.valid_tfrecord_path)

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

        checkpoint.save(self.model_save_dir)

    def _ensure_dir_exist(self, _dir):
        if not tf.io.gfile.exists(_dir):
            tf.io.gfile.makedirs(_dir)

    def _create_optimizer(self):
        return create_optimizer(
            init_lr=self.init_lr,
            num_train_steps=self.total_train_steps,
            num_warmup_steps=int(self.total_train_steps * self.warmup_steps_ratio),
            end_lr=self.end_lr,
            optimizer_type='adamw'
        )

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


# Global Variables #####

# task
TASK_NAME = 'mrc'

# tfrecord
MRC_ALL_TFRECORD_PATH = 'datasets/tfrecord_datasets/mrc_all.tfrecord'
MRC_TRAIN_TFRECORD_PATH = 'datasets/tfrecord_datasets/mrc_train.tfrecord'
MRC_VALID_TFRECORD_PATH = 'datasets/tfrecord_datasets/mrc_valid.tfrecord'

# tfrecord desc
MRC_ALL_TFRECORD_DESC_PATH = 'datasets/tfrecord_datasets/mrc_all_meta_data.json'
MRC_TRAIN_TFRECORD_DESC_PATH = 'datasets/tfrecord_datasets/mrc_train_meta_data.json'
MRC_VALID_TFRECORD_DESC_PATH = 'datasets/tfrecord_datasets/mrc_valid_meta_data.json'

# model
MODEL_SAVE_DIR = './saved_models'
TENSORBOARD_LOG_DIR = './logs/mrc-logs'


def get_model_params():
    # load tfrecord description
    with tf.io.gfile.GFile(MRC_ALL_TFRECORD_DESC_PATH, mode='r') as reader:
        desc = json.load(reader)
    reader.close()

    return collections.defaultdict(
        lambda: None,
        task_name=TASK_NAME,
        distribution_strategy='one_device',
        epochs=15,
        max_seq_len=desc['max_seq_length'],
        total_features=desc['train_data_size'],
        model_save_dir=MODEL_SAVE_DIR,
        tfrecord_path=MRC_ALL_TFRECORD_PATH,
        train_tfrecord_path=MRC_TRAIN_TFRECORD_PATH,
        valid_tfrecord_path=MRC_VALID_TFRECORD_PATH,
        init_lr=1e-4,
        end_lr=0.0,
        warmup_steps_ratio=0.1,
        valid_data_ratio=0.1,
        time_prefix=time.strftime('%Y_%m_%d', time.localtime()),
        enable_tensorboard=True,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR
    )


def main():
    logging.set_verbosity(logging.INFO)
    task = MRCTask(
        get_model_params(),
        use_pretrain=True,
        batch_size=32,
        inference_type=None
    )
    return task


if __name__ == '__main__':
    task = main()
    task.train()
