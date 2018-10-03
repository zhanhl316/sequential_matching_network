#!/bin/env python 
#-*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import codecs
import mds_inputs
import mds_hparams
import mds_metrics
import mds_model
import sys

tf.logging.set_verbosity(tf.logging.DEBUG)

tf.flags.DEFINE_string('test_file', './data/task5.tfrecords', 'Path to the test TFrecords file')
tf.flags.DEFINE_string('model_dir', None, 'Dictionary to load model checkpoins from')
tf.flags.DEFINE_integer('loglevel', 20, 'Tensorflow Log Level')
tf.flags.DEFINE_integer('test_batch_size', 1000, 'the batch size of test ')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if not FLAGS.model_dir:
    print('You must specify a model dictionary')
    print('Usage: python ./mds_test.py --model_dir = xxx')
    sys.exit(1)


if __name__=='__main__':
    hparams = mds_hparams.create_hparams()
    model_fn = mds_model.create_model_fn(hparams)

    estimator = tf.contrib.learn.Estimator(
        model_fn = model_fn,
        model_dir = FLAGS.model_dir,
        config = tf.contrib.learn.RunConfig())

    input_fn_test = mds_inputs.create_input_fn(
        mode  = tf.contrib.learn.ModeKeys.EVAL,
        input_files = [FLAGS.test_file],
        batch_size = FLAGS.test_batch_size,
        num_epochs = 1)

    eval_metrics = mds_metrics.create_evaluation_metrics()
    estimator.evaluate(
            input_fn = input_fn_test, steps = None, metrics = eval_metrics)



