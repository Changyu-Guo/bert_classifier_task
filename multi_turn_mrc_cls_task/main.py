# -*- coding: utf - 8 -*-

import os
import json
import time

import tensorflow as tf
from absl import logging

from optimization import create_optimizer
from utils.distribu_utils import get_distribution_strategy
from utils.distribu_utils import get_strategy_scope
from multi_turn_mrc_cls_task.create_model import create_model
from multi_turn_mrc_cls_task.input_pipeline import read_and_batch_from_tfrecord
from multi_turn_mrc_cls_task.input_pipeline import map_data_to_model


class MultiTurnMRCCLSTask:

    def __init__(
            self,
            kwargs,
            use_pretrain=None,
            batch_size=None,
            inference_model_dir=None
    ):

        # param check
        if use_pretrain is None:
            raise ValueError('Param use_pretrain must be passed')

        self.use_pretrain = use_pretrain
        self.batch_size = batch_size
        self.inference_model_dir = inference_model_dir

        self.task_name = kwargs['task_name']

        # data
        self.train_tfrecord_file_path = kwargs['train_tfrecord_file_path']
        self.valid_tfrecord_file_path = kwargs['valid_tfrecord_file_path']
        self.train_tfrecord_meta_path = kwargs['train_tfrecord_meta_path']
        self.valid_tfrecord_meta_path = kwargs['valid_tfrecord_meta_path']

        # data process
        self.max_seq_len = kwargs['max_seq_len']

        # model and tokenizer
        self.vocab_file_path = kwargs['vocab_file_path']

        # optimizer
        self.init_lr = kwargs['init_lr']
        self.end_lr = kwargs['end_lr']
        self.warmup_steps_ratio = kwargs['warmup_steps_ratio']

        # train
        self.epochs = kwargs['epochs']
        self.distribution_strategy = kwargs['distribution_strategy']
        self.enable_checkpointing = kwargs['enable_checkpointing']
        self.enable_tensorboard = kwargs['enable_tensorboard']
        self.enable_early_stopping = kwargs['enable_early_stopping']

        # output file
        self.model_save_dir = kwargs['model_save_dir']
        self.tensorboard_log_dir = kwargs['tensorboard_log_dir']

        # inference
        self.inference_results_save_dir = kwargs['inference_results_save_dir']
        self.predict_batch_size = kwargs['predict_batch_size']
        self.predict_threshold = kwargs['predict_threshold']

        self.distribution_strategy = get_distribution_strategy(self.distribution_strategy, num_gpus=1)

    def train(self):
        with tf.io.gfile.GFile(self.train_tfrecord_meta_path, mode='r') as reader:
            self.train_meta_data = json.load(reader)
        reader.close()

        train_data_size = self.train_meta_data['data_size']
        # for train
        self.steps_per_epoch = int(train_data_size // self.batch_size) + 1
        # for warmup
        self.total_train_steps = self.steps_per_epoch * self.epochs

        # load tfrecord and transform
        train_dataset = read_and_batch_from_tfrecord(
            filepath=self.train_tfrecord_file_path,
            max_seq_len=self.max_seq_len,
            repeat=True,
            shuffle=True,
            batch_size=self.batch_size
        )
        valid_dataset = read_and_batch_from_tfrecord(
            filepath=self.valid_tfrecord_meta_path,
            max_seq_len=self.max_seq_len,
            repeat=False,
            shuffle=False,
            batch_size=self.batch_size
        )

        train_dataset = train_dataset.map(
            map_data_to_model,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        valid_dataset = valid_dataset.map(
            map_data_to_model,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # 在 distribution strategy scope 下定义:
        #   1. model
        #   2. optimizer
        #   3. load checkpoint
        #   4. compile
        with get_strategy_scope(self.distribution_strategy):

            model = create_model(
                is_train=True,
                use_pretrain=self.use_pretrain
            )
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
                metrics=tf.keras.metrics.BinaryAccuracy()
            )

        callbacks = self._create_callbacks()

        print(model.evaluate(valid_dataset, verbose=0, return_dict=True))
        model.fit(
            train_dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
            validation_data=valid_dataset
        )

        # 保存最后一个 epoch 的模型
        checkpoint.save(self.model_save_dir)

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
                1. checkpoint
                2. summary / tensorboard
                3. earlyStopping
        """
        callbacks = []
        if self.enable_checkpointing:
            ckpt_path = os.path.join(self.model_save_dir, 'ckpt-{epoch:02d}.ckpt')
            # 只保留在 valid dataset 上表现最好的结果
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

        if self.enable_early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    patience=2,
                    restore_best_weights=True
                )
            )

        return callbacks

    def predict_tfrecord(self, tfrecord_path):
        dataset = read_and_batch_from_tfrecord(
            filepath=tfrecord_path,
            max_seq_len=self.max_seq_len,
            repeat=False,
            shuffle=False,
            batch_size=self.predict_batch_size
        )

        with get_strategy_scope(self.distribution_strategy):
            model = create_model(is_train=False, use_pretrain=False)
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(
                tf.train.latest_checkpoint(
                    self.inference_model_dir
                )
            )
            logging.info('Loading checkpoint {} from {}'.format(
                tf.train.latest_checkpoint(self.inference_model_dir),
                self.inference_model_dir
            ))

        for index, data in enumerate(dataset):
            unique_ids = data.pop('unique_ids')
            example_indices = data.pop('example_indices')
            model_output = model.predict(map_data_to_model(data))
            batch_probs = model_output['probs']

            break

    def generate_predict_item(self, unique_ids, example_indices):
        pass


# Global Variables ############

# 任务名称
TASK_NAME = 'multi_turn_mrc_cls_task'

# TFRecord
TRAIN_TFRECORD_FILE_PATH = 'datasets/tfrecords/train.tfrecord'
VALID_TFRECORD_FILE_PATH = 'datasets/tfrecords/valid.tfrecord'

# tfrecord meta data
TRAIN_TFRECORD_META_PATH = 'datasets/tfrecords/train_meta.json'
VALID_TFRECORD_META_PATH = 'datasets/tfrecords/valid_meta.json'

MODEL_SAVE_DIR = 'saved_models'
TENSORBOARD_LOG_DIR = 'logs'
INFERENCE_RESULTS_SAVE_DIR = 'results'

VOCAB_FILE_PATH = '../vocabs/bert-base-chinese-vocab.txt'
MAX_SEQ_LEN = 165
PREDICT_BATCH_SIZE = 128
PREDICT_THRESHOLD = 0.5

# train relate
LEARNING_RATE = 3e-5


def get_model_params():
    return dict(
        task_name=TASK_NAME,
        distribution_strategy='one_device',
        epochs=10,
        predict_batch_size=PREDICT_BATCH_SIZE,
        model_save_dir=MODEL_SAVE_DIR,
        train_tfrecord_file_path=TRAIN_TFRECORD_FILE_PATH,
        valid_tfrecord_file_path=VALID_TFRECORD_FILE_PATH,
        train_tfrecord_meta_path=TRAIN_TFRECORD_META_PATH,
        valid_tfrecord_meta_path=VALID_TFRECORD_META_PATH,
        vocab_file_path=VOCAB_FILE_PATH,
        max_seq_len=MAX_SEQ_LEN,
        init_lr=LEARNING_RATE,
        end_lr=0.0,
        warmup_steps_ratio=0.1,
        enable_checkpointing=False,  # Notice 开启此选项可能会存储大量的 Checkpoint ####
        enable_tensorboard=True,
        enable_early_stopping=True,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        inference_results_save_dir=INFERENCE_RESULTS_SAVE_DIR,
        predict_threshold=PREDICT_THRESHOLD
    )


def main():
    logging.set_verbosity(logging.INFO)
    task = MultiTurnMRCCLSTask(
        get_model_params(),
        use_pretrain=True,
        batch_size=48,
        inference_model_dir='saved_models/version_1'
    )
    return task


if __name__ == '__main__':
    task = main()
    task.predict_tfrecord('datasets/tfrecords/valid.tfrecord')
