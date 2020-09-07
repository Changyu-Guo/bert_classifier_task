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
import tokenization
from inputs_pipeline import read_and_batch_from_squad_tfrecord
from inputs_pipeline import map_data_to_mrc_task
from inputs_pipeline import map_data_to_mrc_predict_task
from squad_processor import FeatureWriter
from squad_processor import read_squad_examples
from squad_processor import convert_examples_to_features
from squad_processor import generate_train_tf_record_from_json_file
from squad_processor import generate_valid_tf_record_from_json_file
from squad_processor import postprocess_output
from squad_processor import write_to_json_files


class MRCTask:

    def __init__(self, kwargs, use_pretrain=None, use_prev_record=False, batch_size=None, inference_type=None):

        # param check
        if use_pretrain is None:
            raise ValueError('Param use_pretrain must be passed')

        self.batch_size = batch_size
        self.use_pretrain = use_pretrain
        self.inference_type = inference_type
        self.use_prev_record = use_prev_record

        self.task_name = kwargs['task_name']
        self.distribution_strategy = kwargs['distribution_strategy']
        self.epochs = kwargs['epochs']
        self.predict_batch_size = kwargs['predict_batch_size']

        self.train_input_file_path = kwargs['train_input_file_path']
        self.valid_input_file_path = kwargs['valid_input_file_path']
        self.train_output_file_path = kwargs['train_output_file_path']
        self.valid_output_file_path = kwargs['valid_output_file_path']
        self.predict_valid_output_file_path = kwargs['predict_valid_output_file_path']
        self.train_output_meta_path = kwargs['train_output_meta_path']
        self.valid_output_meta_path = kwargs['valid_output_meta_path']
        self.predict_valid_output_meta_path = kwargs['predict_valid_output_meta_path']

        self.vocab_file_path = kwargs['vocab_file_path']

        self.max_seq_len = kwargs['max_seq_len']
        self.max_query_len = kwargs['max_query_len']
        self.doc_stride = kwargs['doc_stride']

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

        self.n_best_size = kwargs['n_best_size']
        self.max_answer_len = kwargs['max_answer_len']
        self.null_score_diff_threshold = kwargs['null_score_diff_threshold']
        self.inference_results_save_dir = kwargs['inference_results_save_dir']

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

        # convert examples to tfrecord or load
        if self.use_prev_record:

            with tf.io.gfile.GFile(self.train_output_meta_path, mode='r') as reader:
                self.train_meta_data = json.load(reader)
            reader.close()

            with tf.io.gfile.GFile(self.valid_output_meta_path, mode='r') as reader:
                self.valid_meta_data = json.load(reader)
            reader.close()

        else:

            self.train_meta_data = generate_train_tf_record_from_json_file(
                input_file_path=self.train_input_file_path,
                vocab_file_path=self.vocab_file_path,
                output_path=self.train_output_file_path,
                max_seq_length=self.max_seq_len,
                do_lower_case=True,
                max_query_length=self.max_query_len,
                doc_stride=self.doc_stride,
                version_2_with_negative=False
            )
            with tf.io.gfile.GFile(self.train_output_meta_path, mode='w') as writer:
                writer.write(json.dumps(self.train_meta_data, ensure_ascii=False, indent=2))
            writer.close()

            self.valid_meta_data = generate_valid_tf_record_from_json_file(
                input_file_path=self.valid_input_file_path,
                vocab_file_path=self.vocab_file_path,
                output_path=self.valid_output_file_path,
                max_seq_length=self.max_seq_len,
                do_lower_case=True,
                max_query_length=self.max_query_len,
                doc_stride=self.doc_stride,
                version_2_with_negative=False,
                batch_size=None,
                is_training=True
            )
            with tf.io.gfile.GFile(self.valid_output_meta_path, mode='w') as writer:
                writer.write(json.dumps(self.valid_meta_data, ensure_ascii=False, indent=2))
            writer.close()

        train_data_size = self.train_meta_data['train_data_size']
        # for train
        self.steps_per_epoch = int(train_data_size // self.batch_size)
        # for warmup
        self.total_train_steps = self.steps_per_epoch * self.epochs

        # load tfrecord and transform
        train_dataset = read_and_batch_from_squad_tfrecord(
            filename=self.train_output_file_path,
            max_seq_len=self.max_seq_len,
            is_training=True,
            repeat=True,
            batch_size=self.batch_size
        )
        valid_dataset = read_and_batch_from_squad_tfrecord(
            filename=self.valid_output_file_path,
            max_seq_len=self.max_seq_len,
            is_training=True,
            repeat=False,
            batch_size=self.batch_size
        )

        train_dataset = train_dataset.map(
            map_data_to_mrc_task,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        valid_dataset = valid_dataset.map(
            map_data_to_mrc_task,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

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
                        from_logits=True
                    ),
                    tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    )
                ],
                loss_weights=[0.5, 0.5]
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

    def predict_output(self):
        with get_strategy_scope(self.distribution_strategy):
            model = create_mrc_model(is_train=False, use_pretrain=False)
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(tf.train.latest_checkpoint(self.model_save_dir))

        tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file_path, do_lower_case=True
        )

        valid_examples = read_squad_examples(
            input_file=self.valid_input_file_path,
            is_training=False,
            version_2_with_negative=False
        )

        valid_writer = FeatureWriter(
            filename=self.predict_valid_output_file_path,
            is_training=False
        )

        valid_features = []

        def _append_feature(feature, is_padding):
            if not is_padding:
                valid_features.append(feature)
            valid_writer.process_feature(feature)

        dataset_size = convert_examples_to_features(
            examples=valid_examples,
            tokenizer=tokenizer,
            max_seq_length=self.max_seq_len,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_len,
            is_training=False,
            output_fn=_append_feature,
            batch_size=self.predict_batch_size
        )
        valid_writer.close()

        num_steps = int(dataset_size // self.predict_batch_size)

        dataset = read_and_batch_from_squad_tfrecord(
            self.predict_valid_output_file_path,
            max_seq_len=self.max_seq_len,
            is_training=False,
            repeat=False,
            batch_size=self.predict_batch_size
        )

        # predict
        all_results = []
        for data in dataset:
            unique_ids = data.pop('unique_ids')
            start_logits, end_logits = model.predict(map_data_to_mrc_predict_task(data))

            for result in self.get_raw_results(dict(
                unique_ids=unique_ids,
                start_logits=start_logits,
                end_logits=end_logits
            )):
                all_results.append(result)

        all_predictions, all_nbest_json = postprocess_output(
            all_examples=valid_examples,
            all_features=valid_features,
            all_results=all_results,
            n_best_size=self.n_best_size,
            max_answer_length=self.max_answer_len,
            do_lower_case=True,
            verbose=False
        )

        self.dump_to_files(all_predictions, all_nbest_json)

    def get_raw_results(self, predictions):
        RawResult = collections.namedtuple(
            'RawResult',
            ['unique_id', 'start_logits', 'end_logits']
        )
        for unique_id, start_logits, end_logits in zip(
            predictions['unique_ids'],
            predictions['start_logits'],
            predictions['end_logits']
        ):
            yield RawResult(
                unique_id=unique_id.numpy(),
                start_logits=start_logits.tolist(),
                end_logits=end_logits.tolist()
            )

    def dump_to_files(self, all_predictions, all_nbest_json):
        output_prediction_file = os.path.join(self.inference_results_save_dir, 'predictions.json')
        output_nbest_file = os.path.join(self.inference_results_save_dir, 'nbest_predictions.json')

        write_to_json_files(all_predictions, output_prediction_file)
        write_to_json_files(all_nbest_json, output_nbest_file)


# Global Variables #####

# task
TASK_NAME = 'mrc'

# tfrecord
MRC_TRAIN_INPUT_FILE_PATH = 'datasets/preprocessed_datasets/mrc_train.json'
MRC_VALID_INPUT_FILE_PATH = 'datasets/preprocessed_datasets/mrc_valid.json'

# model
MODEL_SAVE_DIR = './saved_models'
TENSORBOARD_LOG_DIR = './logs/mrc-logs'

TRAIN_OUTPUT_FILE_PATH = 'datasets/tfrecord_datasets/mrc_train.tfrecord'
VALID_OUTPUT_FILE_PATH = 'datasets/tfrecord_datasets/mrc_valid.tfrecord'
PREDICT_VALID_OUTPUT_FILE_PATH = 'datasets/tfrecord_datasets/mrc_predict_valid.tfrecord'
TRAIN_OUTPUT_META_PATH = 'datasets/tfrecord_datasets/mrc_train_meta.json'
VALID_OUTPUT_META_PATH = 'datasets/tfrecord_datasets/mrc_valid_meta.json'
PREDICT_VALID_OUTPUT_META_PATH = 'datasets/tfrecord_datasets/mrc_predict_valid_meta.json'

VOCAB_FILE_PATH = 'vocab.txt'

MAX_SEQ_LEN = 200
MAX_QUERY_LEN = 32
DOC_STRIDE = 128

PREDICT_BATCH_SIZE = 128

# inference
N_BEST_SIZE = 20
MAX_ANSWER_LENGTH = 30
NULL_SCORE_DIFF_THRESHOLD = 0.0

INFERENCE_RESULTS_SAVE_DIR = 'inference_results/mrc_results'


def get_model_params():

    return collections.defaultdict(
        lambda: None,
        task_name=TASK_NAME,
        distribution_strategy='one_device',
        epochs=15,
        predict_batch_size=PREDICT_BATCH_SIZE,
        model_save_dir=MODEL_SAVE_DIR,
        train_input_file_path=MRC_TRAIN_INPUT_FILE_PATH,
        valid_input_file_path=MRC_VALID_INPUT_FILE_PATH,
        train_output_file_path=TRAIN_OUTPUT_FILE_PATH,
        valid_output_file_path=VALID_OUTPUT_FILE_PATH,
        predict_valid_output_file_path=PREDICT_VALID_OUTPUT_FILE_PATH,
        train_output_meta_path=TRAIN_OUTPUT_META_PATH,
        valid_output_meta_path=VALID_OUTPUT_META_PATH,
        predict_valid_output_meta_path=PREDICT_VALID_OUTPUT_META_PATH,
        vocab_file_path=VOCAB_FILE_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_query_len=MAX_QUERY_LEN,
        doc_stride=DOC_STRIDE,
        init_lr=1e-5,
        end_lr=0.0,
        warmup_steps_ratio=0.1,
        valid_data_ratio=0.1,
        time_prefix=time.strftime('%Y_%m_%d', time.localtime()),
        enable_tensorboard=True,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        n_best_size=N_BEST_SIZE,
        max_answer_len=MAX_ANSWER_LENGTH,
        null_score_diff_threshold=NULL_SCORE_DIFF_THRESHOLD,
        inference_results_save_dir=INFERENCE_RESULTS_SAVE_DIR
    )


def main():
    logging.set_verbosity(logging.INFO)
    task = MRCTask(
        get_model_params(),
        use_pretrain=True,
        use_prev_record=False,
        batch_size=32,
        inference_type=None
    )
    return task


if __name__ == '__main__':
    task = main()
    task.train()
