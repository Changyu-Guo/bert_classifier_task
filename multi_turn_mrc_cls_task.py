# -*- coding: utf - 8 -*-

import os
import json
import time
import collections

import numpy as np
import tensorflow as tf
from absl import logging

import tokenization
from optimization import create_optimizer
from create_models import create_binary_cls_model
from utils.distribu_utils import get_distribution_strategy
from utils.distribu_utils import get_strategy_scope
from sklearn.metrics import precision_recall_fscore_support
from data_processors.bi_cls_data_processor_s1 import read_train_examples_from_init_train
from data_processors.bi_cls_data_processor_s1 import read_valid_examples_from_init_train
from data_processors.bi_cls_data_processor_s1 import convert_examples_to_features
from data_processors.bi_cls_data_processor_s1 import FeatureWriter
from data_processors.bi_cls_data_processor_s1 import generate_tfrecord_from_json_file
from data_processors.bi_cls_data_processor_s1 import postprocess_valid_output
from data_processors.bi_cls_data_processor_s1 import postprocess_train_output
from data_processors.inputs_pipeline import read_and_batch_from_bi_cls_record
from data_processors.inputs_pipeline import map_data_to_bi_cls_task


class MultiTurnMRCCLSTask:

    def __init__(
            self,
            kwargs,
            use_pretrain=None,
            use_prev_record=False,
            batch_size=None,
            inference_model_dir=None
    ):

        # param check
        if use_pretrain is None:
            raise ValueError('Param use_pretrain must be passed')

        self.batch_size = batch_size
        self.use_pretrain = use_pretrain
        self.use_prev_record = use_prev_record
        self.inference_model_dir = inference_model_dir

        self.task_name = kwargs['task_name']

        # data
        self.train_input_file_path = kwargs['train_input_file_path']
        self.valid_input_file_path = kwargs['valid_input_file_path']
        self.train_output_file_path = kwargs['train_output_file_path']
        self.valid_output_file_path = kwargs['valid_output_file_path']
        self.predict_train_output_file_path = kwargs['predict_train_output_file_path']
        self.predict_valid_output_file_path = kwargs['predict_valid_output_file_path']
        self.train_output_meta_path = kwargs['train_output_meta_path']
        self.valid_output_meta_path = kwargs['valid_output_meta_path']

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

        # output file
        self.time_prefix = kwargs['time_prefix']
        self.model_save_dir = os.path.join(
            kwargs['model_save_dir'],
            self.time_prefix + '_' + self.task_name + '_' + str(self.epochs)
        )
        self.tensorboard_log_dir = os.path.join(
            kwargs['tensorboard_log_dir'],
            self.time_prefix + '_' + self.task_name + '_' + str(self.epochs)
        )

        # inference
        self.inference_results_save_dir = kwargs['inference_results_save_dir']
        self.predict_batch_size = kwargs['predict_batch_size']
        self.predict_threshold = kwargs['predict_threshold']

        # 如果使用之前生成的 tfrecord 文件，则必须有：
        # 1. tfrecord 文件本身
        # 2. tfrecord 文件的描述文件
        if use_prev_record:
            if not (kwargs['train_output_file_path'] or kwargs['valid_output_file_path']):
                raise ValueError(
                    'Train output file path and valid output file path '
                    'must be set when use prev record is True'
                )

            if not (kwargs['train_output_meta_path'] or kwargs['valid_output_meta_path']):
                raise ValueError(
                    'Train output meta path and valid output mata path'
                    'must be set when use prev record is True'
                )

        self.distribution_strategy = get_distribution_strategy(self.distribution_strategy, num_gpus=1)

    def train(self):
        # 创建文件保存目录
        self._ensure_dir_exist(self.model_save_dir)
        self._ensure_dir_exist(self.tensorboard_log_dir)

        # convert examples to tfrecord or load
        if self.use_prev_record:

            with tf.io.gfile.GFile(self.train_output_meta_path, mode='r') as reader:
                self.train_meta_data = json.load(reader)
            reader.close()

            with tf.io.gfile.GFile(self.valid_output_meta_path, mode='r') as reader:
                self.valid_meta_data = json.load(reader)
            reader.close()

        else:

            self.train_meta_data = generate_tfrecord_from_json_file(
                input_file_path=self.train_input_file_path,
                vocab_file_path=self.vocab_file_path,
                output_file_path=self.train_output_file_path,
                max_seq_len=self.max_seq_len,
                is_train=True,
                return_examples_and_features=False
            )
            with tf.io.gfile.GFile(self.train_output_meta_path, mode='w') as writer:
                writer.write(json.dumps(self.train_meta_data, ensure_ascii=False, indent=2))
            writer.close()

            self.valid_meta_data = generate_tfrecord_from_json_file(
                input_file_path=self.valid_input_file_path,
                vocab_file_path=self.vocab_file_path,
                output_file_path=self.valid_output_file_path,
                max_seq_len=self.max_seq_len,
                is_train=False,
                return_examples_and_features=False
            )
            with tf.io.gfile.GFile(self.valid_output_meta_path, mode='w') as writer:
                writer.write(json.dumps(self.valid_meta_data, ensure_ascii=False, indent=2))
            writer.close()

        train_data_size = self.train_meta_data['data_size']
        # for train
        self.steps_per_epoch = int(train_data_size // self.batch_size) + 1
        # for warmup
        self.total_train_steps = self.steps_per_epoch * self.epochs

        # load tfrecord and transform
        train_dataset = read_and_batch_from_bi_cls_record(
            filename=self.train_output_file_path,
            max_seq_len=self.max_seq_len,
            repeat=True,
            shuffle=True,
            batch_size=self.batch_size
        )
        valid_dataset = read_and_batch_from_bi_cls_record(
            filename=self.valid_output_file_path,
            max_seq_len=self.max_seq_len,
            repeat=False,
            shuffle=False,
            batch_size=self.batch_size
        )

        train_dataset = train_dataset.map(
            map_data_to_bi_cls_task,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        valid_dataset = valid_dataset.map(
            map_data_to_bi_cls_task,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # 在 distribution strategy scope 下定义:
        #   1. model
        #   2. optimizer
        #   3. load checkpoint
        #   4. compile
        with get_strategy_scope(self.distribution_strategy):

            model = create_binary_cls_model(
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
            validation_data=valid_dataset,
            class_weight={
                0: 1,
                1: 3
            }
        )

        # 保存最后一个 epoch 的模型
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
                2. summary / tensorboard (可选)
                3. earlyStopping (可选)
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

        return callbacks

    def predict_train_data(self):
        self._ensure_dir_exist(self.inference_results_save_dir)

        # 以 valid dataset 的形式构建 train dataset
        # 用于 train 最后一个 step
        generate_tfrecord_from_json_file(
            input_file_path=self.train_input_file_path,
            vocab_file_path=self.vocab_file_path,
            output_file_path=self.predict_train_output_file_path,
            max_seq_len=self.max_seq_len,
            is_train=False,
            return_examples_and_features=True
        )

        dataset = read_and_batch_from_bi_cls_record(
            filename=self.predict_train_output_file_path,
            max_seq_len=self.max_seq_len,
            repeat=False,
            shuffle=False,
            batch_size=self.predict_batch_size
        )

        with get_strategy_scope(self.distribution_strategy):
            model = create_binary_cls_model(
                is_train=False,
                use_pretrain=False
            )
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(
                tf.train.latest_checkpoint(
                    self.inference_model_dir
                )
            )
            logging.info('Loading checkpoint from {}'.format(
                tf.train.latest_checkpoint(
                    self.inference_model_dir
                )
            ))

        origin_is_valid = []
        pred_is_valid = []
        batched_origin_is_valid = []
        batched_pred_is_valid = []

        for index, data in enumerate(dataset):
            x, y = map_data_to_bi_cls_task(data)

            batch_probs = model.predict(x)

            batch_probs = tf.where(batch_probs > self.predict_threshold, 1, 0)

            batch_probs = batch_probs.numpy().flatten().tolist()

            origin_is_valid.extend(y.numpy().tolist())
            pred_is_valid.extend(batch_probs)

            print(index)

        def chunks(ls, n):
            for i in range(0, len(ls), n):
                yield ls[i: i + n]

        for chunk in chunks(origin_is_valid, 53):
            batched_origin_is_valid.append(chunk)

        for chunk in chunks(pred_is_valid, 53):
            batched_pred_is_valid.append(chunk)

        postprocess_train_output(
            batched_origin_is_valid=batched_origin_is_valid,
            batched_pred_is_valid=batched_pred_is_valid,
            results_save_path=os.path.join(self.inference_results_save_dir, 'train_results.json')
        )

    def predict_valid_data(self):

        dataset = read_and_batch_from_bi_cls_record(
            filename=self.predict_valid_output_file_path,
            max_seq_len=self.max_seq_len,
            repeat=False,
            shuffle=False,
            batch_size=self.predict_batch_size
        )

        with get_strategy_scope(self.distribution_strategy):
            model = create_binary_cls_model(
                is_train=False,
                use_pretrain=False
            )
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(
                tf.train.latest_checkpoint(
                    self.inference_model_dir
                )
            )
            logging.info('Loading checkpoint from {}'.format(
                tf.train.latest_checkpoint(
                    self.inference_model_dir
                )
            ))

        origin_is_valid = []
        pred_is_valid = []
        batched_origin_is_valid = []
        batched_pred_is_valid = []

        for index, data in enumerate(dataset):
            x, y = map_data_to_bi_cls_task(data)

            batch_probs = model.predict(x)

            batch_probs = tf.where(batch_probs > self.predict_threshold, 1, 0)

            batch_probs = batch_probs.numpy().flatten().tolist()

            origin_is_valid.extend(y.numpy().tolist())
            pred_is_valid.extend(batch_probs)

            print(index)

        def chunks(ls, n):
            for i in range(0, len(ls), n):
                yield ls[i: i + n]

        for chunk in chunks(origin_is_valid, 53):
            batched_origin_is_valid.append(chunk)

        for chunk in chunks(pred_is_valid, 53):
            batched_pred_is_valid.append(chunk)

        postprocess_valid_output(
            batched_origin_is_valid=batched_origin_is_valid,
            batched_pred_is_valid=batched_pred_is_valid,
            results_save_path=os.path.join(self.inference_results_save_dir, 'valid_results.json')
        )


# Global Variables ############

# 任务名称
TASK_NAME = 'multi_turn_mrc_cls_task'

# 原始数据加载路径
# 当前任务的原始数据应该是 init-train
TRAIN_INPUT_FILE_PATH = 'datasets/raw_datasets/init-train-train.json'
VALID_INPUT_FILE_PATH = 'datasets/raw_datasets/init-train-valid.json'

# 对原始数据处理后的 TFRecord
TRAIN_OUTPUT_FILE_PATH = 'datasets/tfrecord_datasets/bi_cls_s1_train.tfrecord'
VALID_OUTPUT_FILE_PATH = 'datasets/tfrecord_datasets/bi_cls_s1_valid.tfrecord'
PREDICT_TRAIN_OUTPUT_FILE_PATH = 'datasets/tfrecord_datasets/bi_cls_s1_predict_train.tfrecord'
PREDICT_VALID_OUTPUT_FILE_PATH = 'datasets/tfrecord_datasets/bi_cls_s1_predict_valid.tfrecord'

# tfrecord meta data
TRAIN_OUTPUT_META_PATH = 'datasets/tfrecord_datasets/bi_cls_s1_train_meta.json'
VALID_OUTPUT_META_PATH = 'datasets/tfrecord_datasets/bi_cls_s1_valid_meta.json'

# save relate
MODEL_SAVE_DIR = 'saved_models/bi_cls_s1_models'
TENSORBOARD_LOG_DIR = 'logs/bi-cls-s1-logs'

# tokenize
VOCAB_FILE_PATH = 'vocabs/bert-base-chinese-vocab.txt'

# dataset process relate
MAX_SEQ_LEN = 165
PREDICT_BATCH_SIZE = 10000
PREDICT_THRESHOLD = 0.5

# train relate
LEARNING_RATE = 3e-5

# inference relate
INFERENCE_RESULTS_SAVE_DIR = 'inference_results/bi_cls_s1_results'


def get_model_params():
    return collections.defaultdict(
        lambda: None,
        task_name=TASK_NAME,
        distribution_strategy='one_device',
        epochs=4,
        predict_batch_size=PREDICT_BATCH_SIZE,
        model_save_dir=MODEL_SAVE_DIR,
        train_input_file_path=TRAIN_INPUT_FILE_PATH,
        valid_input_file_path=VALID_INPUT_FILE_PATH,
        train_output_file_path=TRAIN_OUTPUT_FILE_PATH,
        valid_output_file_path=VALID_OUTPUT_FILE_PATH,
        predict_train_output_file_path=PREDICT_TRAIN_OUTPUT_FILE_PATH,
        predict_valid_output_file_path=PREDICT_VALID_OUTPUT_FILE_PATH,
        train_output_meta_path=TRAIN_OUTPUT_META_PATH,
        valid_output_meta_path=VALID_OUTPUT_META_PATH,
        vocab_file_path=VOCAB_FILE_PATH,
        max_seq_len=MAX_SEQ_LEN,
        init_lr=LEARNING_RATE,
        end_lr=0.0,
        warmup_steps_ratio=0.1,
        time_prefix=time.strftime('%Y_%m_%d', time.localtime()),  # 年_月_日
        enable_checkpointing=False,  # Notice 开启此选项可能会存储大量的 Checkpoint ####
        enable_tensorboard=True,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        inference_results_save_dir=INFERENCE_RESULTS_SAVE_DIR,
        predict_threshold=PREDICT_THRESHOLD
    )


def bi_cls_s1_main():
    logging.set_verbosity(logging.INFO)
    task = MultiTurnMRCCLSTask(
        get_model_params(),
        use_pretrain=True,
        use_prev_record=True,
        batch_size=48,
        inference_model_dir='gs://leeyu-dataset-public'
    )
    return task


if __name__ == '__main__':
    task = bi_cls_s1_main()
    task.predict_train_data()
