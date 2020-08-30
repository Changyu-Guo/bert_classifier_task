# -*- coding: utf - 8 -*-

import tensorflow as tf


def compute_prf(y_true, y_pred, average='macro'):
    if average == 'micro':
        axis = None
    elif average == 'macro':
        axis = 0
    else:
        raise ValueError('average must as macro or micro')

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    TP = tf.math.count_nonzero(y_pred * y_true, axis=axis)
    FP = tf.math.count_nonzero(y_pred * (y_true - 1), axis=axis)
    FN = tf.math.count_nonzero((y_pred - 1) * y_true, axis=axis)

    TP = TP.numpy()
    FP = FP.numpy()
    FN = FN.numpy()

    P = TP / ((TP + FP) + 1e-100)
    R = TP / ((TP + FN) + 1e-100)
    F = 2 * P * R / ((P + R) + 1e-100)

    return P, R, F
