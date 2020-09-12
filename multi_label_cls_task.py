# -*- coding: utf - 8 -*-

import os
import json
import time
import collections
import tensorflow as tf
import tensorflow_addons as tfa
from absl import logging

from create_models import create_multi_label_cls_model
from optimization import create_optimizer
from utils.distribu_utils import get_distribution_strategy
from utils.distribu_utils import get_strategy_scope
from data_processors.multi_label_cls_data_processor import extract_relations_from_init_train_table
from data_processors.multi_label_cls_data_processor import FeaturesWriter
from data_processors.multi_label_cls_data_processor import read_init_train_examples
from data_processors.multi_label_cls_data_processor import generate_tfrecord_from_json_file
from data_processors.multi_label_cls_data_processor import convert_examples_to_features
from data_processors.multi_label_cls_data_processor import postprocess_output
from data_processors.inputs_pipeline import read_and_batch_from_multi_label_cls_tfrecord
from data_processors.inputs_pipeline import map_data_to_multi_label_cls_train_task
from data_processors.inputs_pipeline import map_data_to_multi_label_cls_predict_task


class CLSTask:

    def __init__(
            self,
            kwargs,
            use_pretrain=True,
            use_prev_record=False,
            batch_size=None,
            inference_model_dir=None
    ):

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

        # model and tokenize
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
        self.predict_batch_size = kwargs['predict_batch_size']
        self.inference_results_save_dir = kwargs['inference_results_save_dir']
        self.predict_threshold = kwargs['predict_threshold']

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
        self._ensure_dir_exist(self.model_save_dir)
        self._ensure_dir_exist(self.tensorboard_log_dir)

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
                max_seq_len=self.max_seq_len
            )
            self.valid_meta_data = generate_tfrecord_from_json_file(
                input_file_path=self.valid_input_file_path,
                vocab_file_path=self.vocab_file_path,
                output_file_path=self.valid_output_file_path,
                max_seq_len=self.max_seq_len
            )
            with tf.io.gfile.GFile(self.train_output_meta_path, mode='w') as writer:
                writer.write(json.dumps(self.train_meta_data, ensure_ascii=False, indent=2))
            writer.close()
            with tf.io.gfile.GFile(self.valid_output_meta_path, mode='w') as writer:
                writer.write(json.dumps(self.valid_meta_data, ensure_ascii=False, indent=2))
            writer.close()

        self.num_labels = self.train_meta_data['num_labels']
        self.train_data_size = self.train_meta_data['data_size']
        self.steps_per_epoch = int(self.train_data_size // self.batch_size) + 1
        self.total_train_steps = self.steps_per_epoch * self.epochs

        train_dataset = read_and_batch_from_multi_label_cls_tfrecord(
            filename=self.train_output_file_path,
            max_seq_len=self.max_seq_len,
            num_labels=self.num_labels,
            shuffle=True,
            repeat=True,
            batch_size=self.batch_size
        )
        valid_dataset = read_and_batch_from_multi_label_cls_tfrecord(
            filename=self.valid_output_file_path,
            max_seq_len=self.max_seq_len,
            num_labels=self.num_labels,
            shuffle=True,
            repeat=False,
            batch_size=self.batch_size
        )

        train_dataset = train_dataset.map(
            map_data_to_multi_label_cls_train_task,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        valid_dataset = valid_dataset.map(
            map_data_to_multi_label_cls_train_task,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # 在 distribution strategy scope 下定义:
        #   1. model
        #   2. optimizer
        #   3. load checkpoint
        #   4. compile
        with get_strategy_scope(self.distribution_strategy):

            model = create_multi_label_cls_model(
                self.num_labels,
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
                loss={
                    'probs': tf.keras.losses.BinaryCrossentropy(from_logits=False)
                },
                metrics=['binary_accuracy']
            )

        callbacks = self._create_callbacks()

        model.fit(
            train_dataset,
            initial_epoch=0,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
            validation_data=valid_dataset
        )

        checkpoint.save(self.model_save_dir)

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

    def predict_train_data(self):
        _, relations, _, _ = extract_relations_from_init_train_table()
        num_labels = len(relations)

        with get_strategy_scope(self.distribution_strategy):
            model = create_multi_label_cls_model(
                num_labels,
                is_train=False,
                use_pretrain=False
            )
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(
                tf.train.latest_checkpoint(
                    self.inference_model_dir
                )
            )
        train_examples = read_init_train_examples(self.train_input_file_path)

        train_writer = FeaturesWriter(
            filename=self.predict_train_output_file_path
        )

        train_features = []

        def _append_feature(feature):
            train_features.append(feature)
            train_writer.process_feature(feature)

        convert_examples_to_features(
            examples=train_examples,
            vocab_file_path=self.vocab_file_path,
            labels=relations,
            max_seq_len=self.max_seq_len,
            output_fn=_append_feature
        )
        train_writer.close()

        train_dataset = read_and_batch_from_multi_label_cls_tfrecord(
            self.predict_train_output_file_path,
            max_seq_len=self.max_seq_len,
            num_labels=num_labels,
            shuffle=False,
            repeat=False,
            batch_size=self.predict_batch_size
        )
        train_results = []
        for index, data in enumerate(train_dataset):
            unique_ids = data.pop('unique_ids')
            model_output = model.predict(
                map_data_to_multi_label_cls_predict_task(data)
            )
            batch_probs = model_output['probs']
            for result in self.generate_predict_item(unique_ids, batch_probs):
                train_results.append(result)

            print(index)

        postprocess_output(
            all_relations=relations,
            all_features=train_features,
            all_results=train_results,
            threshold=self.predict_threshold,
            results_save_path=os.path.join(self.inference_results_save_dir, 'train_results.json')
        )

    def predict_valid_data(self):
        _, relations, _, _ = extract_relations_from_init_train_table()
        num_labels = len(relations)

        with get_strategy_scope(self.distribution_strategy):
            model = create_multi_label_cls_model(
                num_labels,
                is_train=False,
                use_pretrain=False
            )
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(
                tf.train.latest_checkpoint(
                    self.inference_model_dir
                )
            )

        valid_examples = read_init_train_examples(self.valid_input_file_path)

        valid_writer = FeaturesWriter(
            filename=self.predict_valid_output_file_path
        )

        valid_features = []

        def _append_feature(feature):
            valid_features.append(feature)
            valid_writer.process_feature(feature)

        convert_examples_to_features(
            examples=valid_examples,
            vocab_file_path=self.vocab_file_path,
            labels=relations,
            max_seq_len=self.max_seq_len,
            output_fn=_append_feature
        )
        valid_writer.close()

        valid_dataset = read_and_batch_from_multi_label_cls_tfrecord(
            filename=self.predict_valid_output_file_path,
            max_seq_len=self.max_seq_len,
            num_labels=num_labels,
            shuffle=False,
            repeat=False,
            batch_size=self.predict_batch_size
        )
        valid_results = []
        for data in valid_dataset:
            unique_ids = data.pop('unique_ids')
            model_output = model.predict(
                map_data_to_multi_label_cls_predict_task(data)
            )
            batch_probs = model_output['probs']
            for result in self.generate_predict_item(unique_ids, batch_probs):
                valid_results.append(result)

        postprocess_output(
            all_relations=relations,
            all_features=valid_features,
            all_results=valid_results,
            threshold=self.predict_threshold,
            results_save_path=os.path.join(self.inference_results_save_dir, 'valid_results.json')
        )

    def generate_predict_item(self, unique_ids, batch_probs):
        RawResult = collections.namedtuple(
            'RawResult',
            ['unique_id', 'probs']
        )

        for unique_id, probs in zip(unique_ids, batch_probs):
            yield RawResult(
                unique_id=unique_id.numpy(),
                probs=probs
            )


# Global Variables #####

# task
TASK_NAME = 'multi_label_cls'

# raw json path
TRAIN_INPUT_FILE_PATH = 'datasets/preprocessed_datasets/multi_label_cls_train.json'
VALID_INPUT_FILE_PATH = 'datasets/preprocessed_datasets/multi_label_cls_valid.json'

# tfrecord path
TRAIN_OUTPUT_FILE_PATH = 'datasets/tfrecord_datasets/multi_label_cls_train.tfrecord'
VALID_OUTPUT_FILE_PATH = 'datasets/tfrecord_datasets/multi_label_cls_valid.tfrecord'
PREDICT_TRAIN_OUTPUT_FILE_PATH = 'datasets/tfrecord_datasets/multi_label_cls_predict_train.tfrecord'
PREDICT_VALID_OUTPUT_FILE_PATH = 'datasets/tfrecord_datasets/multi_label_cls_predict_valid.tfrecord'

# tfrecord meta data
TRAIN_OUTPUT_META_PATH = 'datasets/tfrecord_datasets/multi_label_cls_train_meta.json'
VALID_OUTPUT_META_PATH = 'datasets/tfrecord_datasets/multi_label_cls_valid_meta.json'

# save relate
MODEL_SAVE_DIR = 'saved_models/multi_label_cls_models'
TENSORBOARD_LOG_DIR = 'logs/multi-label-cls-logs'

# tokenize
VOCAB_FILE_PATH = 'vocabs/bert-base-chinese-vocab.txt'

# dataset process relate
MAX_SEQ_LEN = 120
PREDICT_BATCH_SIZE = 128
PREDICT_THRESHOLD = 0.5

# train relate
LEARNING_RATE = 1e-4

# inference relate
INFERENCE_RESULTS_SAVE_DIR = 'inference_results/multi_label_cls_results'


def get_model_params():
    return collections.defaultdict(
        lambda: None,
        task_name=TASK_NAME,
        distribution_strategy='one_device',
        epochs=20,
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
        time_prefix=time.strftime('%Y_%m_%d', time.localtime()),
        enable_checkpoint=False,
        enable_tensorboard=True,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        inference_results_save_dir=INFERENCE_RESULTS_SAVE_DIR,
        predict_threshold=PREDICT_THRESHOLD
    )


def multi_label_cls_main():
    logging.set_verbosity(logging.INFO)
    task = CLSTask(
        get_model_params(),
        use_pretrain=True,  # Notice ###
        use_prev_record=True,
        batch_size=80,  # Notice ###
        inference_model_dir='saved_models/multi_label_cls_models/epochs_15'
    )
    return task


if __name__ == '__main__':
    task = multi_label_cls_main()
    task.train()
