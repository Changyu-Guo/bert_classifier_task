# -*- coding: utf - 8 -*-

import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, classification_report
import tensorflow_addons as tfa
from custom_metrics import compute_prf

y_true = [
    [1, 0, 1, 0, 0],
    [1, 0, 1, 0, 0]]
y_pred = [
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]]

print(classification_report(y_true, y_pred))
