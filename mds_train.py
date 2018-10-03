#!/bin/env python
#-*- encoding:utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import codecs
import tensorflow as tf
from tensorflow.contrib import learn
import mds_hparams
import mds_inputs
import mds_model
import mds_metrics
from gensim import models


tf.logging.set_verbosity(tf.logging.DEBUG)

tf.flags.DEFINE_string(
    'input_dir', './data', 'Directory containing input data files (xxx.tfrecordes)')
tf.flags.DEFINE_string(
    'model_dir', None, 'Directory to store model checkpoints (defaults to ./runs)')
tf.flags.DEFINE_integer(
    'num_epochs', 10, 'Number of training Epochs. Defaults to indefinite.')
tf.flags.DEFINE_integer(
    'eval_every', 500, 'Evaluate after this many train steps')
tf.flags.DEFINE_integer(
    'checkpoints_steps', 1000, 'save checkpoints of steps')
tf.flags.DEFINE_integer(
    'keep_checkpoint_max', 10, 'max nums of checkpoint for saved')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()



TIMESTAMP = int(time.time())
if FLAGS.model_dir:
    MODEL_DIR = FLAGS.model_dir
else:
    MODEL_DIR = os.path.abspath(os.path.join('./runs', str(TIMESTAMP)))


TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, 'train.tfrecords'))
VALIDATION_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, 'task5.tfrecords'))


def main(unused_argv):
    hparams = mds_hparams.create_hparams()
    #loading the train and validation data
    
    model_fn = mds_model.create_model_fn(hparams)
    #load model

    sess_cfg = tf.ConfigProto(device_count = {'GPU': 0})
    run_cfg = learn.RunConfig(session_config = sess_cfg, 
                              save_checkpoints_steps = FLAGS.checkpoints_steps,
                              keep_checkpoint_max = FLAGS.keep_checkpoint_max,
                              keep_checkpoint_every_n_hours = 10000)

    estimator = learn.Estimator(
        model_fn = model_fn,
        model_dir = MODEL_DIR,
        config = run_cfg)

    # set up logging for predictions
    # log the values in the 'Softmax' tensor with label 'probabilities'
    tensors_to_log = {"probabilities": 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter = 100)


    input_fn_train = mds_inputs.create_input_fn(
        mode = learn.ModeKeys.TRAIN,
        input_files = [TRAIN_FILE],
        batch_size = hparams.batch_size,
        num_epochs = FLAGS.num_epochs)
    
    input_fn_eval = mds_inputs.create_input_fn(
        mode = learn.ModeKeys.EVAL,
        input_files = [VALIDATION_FILE],
        batch_size = hparams.eval_batch_size,
        num_epochs = 1)
   
     # set up eval metrics
    eval_metrics = mds_metrics.create_evaluation_metrics()
    eval_monitor = learn.monitors.ValidationMonitor(
        input_fn = input_fn_eval,
        every_n_steps = FLAGS.eval_every,
        metrics = eval_metrics,
        eval_steps = None,
        early_stopping_metric = 'loss',
        early_stopping_metric_minimize = True,
        early_stopping_rounds = None)
        #early_stopping_rounds = 3)

    # train and evaluate
    estimator.fit(input_fn = input_fn_train,
                  steps = None,
                  max_steps = hparams.max_steps,
                  monitors = [eval_monitor, logging_hook])




if __name__=='__main__':
    tf.app.run()






