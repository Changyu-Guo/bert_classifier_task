# -*- coding: utf - 8 -*-

import os
import sys
import json

import tensorflow as tf
from absl import logging

sys.path.append('..')

from naive_mrc_task.create_model import create_model
from optimization import create_optimizer
from utils.distribu_utils import get_distribution_strategy
from utils.distribu_utils import get_strategy_scope
from naive_mrc_task.input_pipeline import read_and_batch_from_tfrecord
from naive_mrc_task.input_pipeline import map_data_to_model_train
from naive_mrc_task.input_pipeline import map_data_to_model_infer


class MRCTask:

    def __init__(
            self,
            kwargs,
            batch_size=None,
            use_net_pretrain=None
    ):

        # param check
        if use_net_pretrain is None:
            raise ValueError('Param use_pretrain must be passed')

        if not use_net_pretrain:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'

        self.batch_size = batch_size
        self.use_net_pretrain = use_net_pretrain

        self.task_name = kwargs['task_name']

        # data
        self.train_tfrecord_file_path = kwargs['train_tfrecord_file_path']
        self.valid_tfrecord_file_path = kwargs['valid_tfrecord_file_path']
        self.train_tfrecord_meta_path = kwargs['train_tfrecord_meta_path']
        self.valid_tfrecord_meta_path = kwargs['valid_tfrecord_meta_path']

        # data process
        self.max_seq_len = kwargs['max_seq_len']
        self.max_query_len = kwargs['max_query_len']
        self.doc_stride = kwargs['doc_stride']

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
        self.predict_batch_size = kwargs['predict_batch_size']

        self.distribution_strategy = get_distribution_strategy(self.distribution_strategy, num_gpus=1)

    def train(self):

        with tf.io.gfile.GFile(self.train_tfrecord_meta_path, mode='r') as reader:
            self.train_meta_data = json.load(reader)
        reader.close()

        train_data_size = self.train_meta_data['data_size']
        # for train
        self.steps_per_epoch = int(train_data_size // self.batch_size)
        if train_data_size % self.batch_size != 0:
            self.steps_per_epoch += 1
        # for warmup
        self.total_train_steps = self.steps_per_epoch * self.epochs

        # load tfrecord and transform
        train_dataset = read_and_batch_from_tfrecord(
            filepath=self.train_tfrecord_file_path,
            max_seq_len=self.max_seq_len,
            is_training=True,
            repeat=True,
            shuffle=True,
            batch_size=self.batch_size
        )
        valid_dataset = read_and_batch_from_tfrecord(
            filepath=self.valid_tfrecord_file_path,
            max_seq_len=self.max_seq_len,
            is_training=True,  # 这里设置为 True, 使得数据集包含 start / end position
            repeat=False,
            shuffle=False,
            batch_size=self.batch_size
        )

        train_dataset = train_dataset.map(
            map_data_to_model_train,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        valid_dataset = valid_dataset.map(
            map_data_to_model_train,
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
                use_net_pretrain=self.use_net_pretrain
            )
            optimizer = self._create_optimizer()

            # load checkpoint
            checkpoint = tf.train.Checkpoint(
                model=model,
                optimizer=optimizer
            )

            loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            model.compile(
                optimizer=optimizer,
                loss=[loss_func, loss_func],
                loss_weights=[0.5, 0.5],
                metrics=['acc'],
            )

        callbacks = self._create_callbacks()

        # print(model.evaluate(valid_dataset, verbose=0, return_dict=True))
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
                1. checkpoint (重要)
                2. summary / tensorboard (可选)
                3. earlyStopping (重要)
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
                    monitor='val_end_logits_acc',
                    mode='max',
                    patience=6,
                    restore_best_weights=True
                )
            )

        return callbacks

    def predict_tfrecord(self, inference_model_dir, tfrecord_path, save_path):

        dataset = read_and_batch_from_tfrecord(
            filepath=tfrecord_path,
            max_seq_len=self.max_seq_len,
            is_training=False,  # 推断的过程中将 is_training 设置为 False, 不再读取真实答案
            repeat=False,
            shuffle=False,
            batch_size=self.predict_batch_size
        )

        with get_strategy_scope(self.distribution_strategy):
            model = create_model(
                is_train=False, use_net_pretrain=False
            )
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(
                tf.train.latest_checkpoint(
                    inference_model_dir
                )
            )
            logging.info('Restore checkpoint {} from {}'.format(
                tf.train.latest_checkpoint(inference_model_dir),
                inference_model_dir
            ))

        # predict
        all_results = []
        for index, data in enumerate(dataset):
            # (batch_size, 1)
            unique_ids = data.pop('unique_ids')
            example_indices = data.pop('example_indices')
            # (batch_size, seq_len)
            model_output = model.predict(map_data_to_model_infer(data))

            start_logits = model_output['start_logits']
            end_logits = model_output['end_logits']

            for result in self.generate_predict_item(
                unique_ids=unique_ids,
                example_indices=example_indices,
                batch_start_logits=start_logits,
                batch_end_logits=end_logits
            ):
                all_results.append(result)

            print(index)

        with tf.io.gfile.GFile(save_path, mode='w') as writer:
            writer.write(json.dumps(all_results, ensure_ascii=False, indent=2) + '\n')
        writer.close()

    def generate_predict_item(self, unique_ids, example_indices, batch_start_logits, batch_end_logits):
        for unique_id, example_index, start_logits, end_logits in zip(
            unique_ids,
            example_indices,
            batch_start_logits,
            batch_end_logits
        ):
            yield dict(
                unique_id=unique_id.numpy().item(),
                example_index=example_index.numpy().item(),
                start_logits=start_logits.tolist(),
                end_logits=end_logits.tolist()
            )


# Global Variables ############

# task
TASK_NAME = 'mrc'

# TFRecord
TRAIN_TFRECORD_FILE_PATH = 'datasets/version_5/train/first/tfrecords/train.tfrecord'
VALID_TFRECORD_FILE_PATH = 'datasets/version_5/train/first/tfrecords/valid.tfrecord'

# tfrecord meta data
TRAIN_TFRECORD_META_PATH = 'datasets/version_5/train/first/meta/train_meta.json'
VALID_TFRECORD_META_PATH = 'datasets/version_5/train/first/meta/valid_meta.json'

MODEL_SAVE_DIR = 'saved_models/version_5/first/naive_mrc.ckpt'
TENSORBOARD_LOG_DIR = 'logs/version_5'

# dataset process relate
MAX_SEQ_LEN = 200
MAX_QUERY_LEN = 50
DOC_STRIDE = 128
PREDICT_BATCH_SIZE = 128

# train relate
LEARNING_RATE = 1e-4


def get_model_params():
    return dict(
        task_name=TASK_NAME,
        distribution_strategy='one_device',
        epochs=100,
        predict_batch_size=PREDICT_BATCH_SIZE,
        model_save_dir=MODEL_SAVE_DIR,
        train_tfrecord_file_path=TRAIN_TFRECORD_FILE_PATH,
        valid_tfrecord_file_path=VALID_TFRECORD_FILE_PATH,
        train_tfrecord_meta_path=TRAIN_TFRECORD_META_PATH,
        valid_tfrecord_meta_path=VALID_TFRECORD_META_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_query_len=MAX_QUERY_LEN,
        doc_stride=DOC_STRIDE,
        init_lr=LEARNING_RATE,
        end_lr=1e-7,
        warmup_steps_ratio=0.1,
        enable_checkpointing=False,  # Notice 开启此选项可能会存储大量的 Checkpoint ####
        enable_tensorboard=True,
        enable_early_stopping=True,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
    )


def main():
    logging.set_verbosity(logging.INFO)
    task = MRCTask(
        get_model_params(),
        batch_size=40,
        use_net_pretrain=True,
    )
    return task


if __name__ == '__main__':
    task = main()

    # task.predict_tfrecord(
    #     inference_model_dir='saved_models/version_4/',
    #     tfrecord_path='datasets/version_4/inference/last_version_1/second/tfrecords/valid.tfrecord',
    #     save_path='inference_results/version_4/last_version_1/second/raw/valid_results.json'
    # )

    task.train()
