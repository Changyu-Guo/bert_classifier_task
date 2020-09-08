# -*- coding: utf - 8 -*-

import tensorflow as tf


gs = 'gs://leeyu-checkpoint'

print(tf.io.gfile.glob(gs))