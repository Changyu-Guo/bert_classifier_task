# -*- coding: utf - 8 -*-

import tensorflow as tf


def squad_loss_fn(x):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    start_loss = loss_fn(x[0], x[2])
    end_loss = loss_fn(x[1], x[3])

    return (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2.0
