# -*- coding: utf - 8 -*-

import tensorflow as tf
from create_models import create_mrc_model


class MRCTask:

    def __init__(self, kwargs, use_pretrain=None, batch_size=None, inference_type=None):
        if use_pretrain is None or batch_size is None:
            raise ValueError('Param use_pretrain and batch_size must be pass')
        self.batch_size = batch_size
        self.use_pretrain = use_pretrain
        self.inference_type = inference_type

        self.task_name = kwargs['task_name']
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
                loss={
                    'start_logits': tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True,
                        reduction=tf.keras.losses.Reduction.NONE
                    ),
                    'end_logits': tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True,
                        reduction=tf.keras.losses.Reduction.NONE
                    )
                },
                loss_weights=[1.0, 1.0]
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
        max_seq_len=desc['max_seq_len'],
        total_features=desc['train_data_size'],
        model_save_dir=MODEL_SAVE_DIR,
        tfrecord_path=MRC_ALL_TFRECORD_PATH,
        init_lr=1e-4,
        end_lr=0.0,
        warmup_steps_ratio=0.1,
        valid_data_ratio=0.1,
        inference_result_path=os.path.join(
            INFERENCE_RESULTS_DIR,
            time.strftime('%Y_%m_%d', time.localtime()) + '_result.txt'),
        train_tfrecord_path=MRC_TRAIN_TFRECORD_PATH,
        valid_tfrecord_path=MRC_VALID_TFRECORD_PATH,
        enable_tensorboard=True,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR
    )
