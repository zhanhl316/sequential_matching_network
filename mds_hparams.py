#!/bin/env python
#-*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import subprocess
from collections import namedtuple


#Global Configuration
if os.path.exists('./data/vocabulary.txt'):
    _VOCAB_SIZE = subprocess.check_output(
        "wc -l ./data/vocabulary.txt | awk '{print $1}'", shell = True)
    _VOCAB_SIZE = int(_VOCAB_SIZE.strip('\n')) + 1
    tf.flags.DEFINE_integer('vocab_size', _VOCAB_SIZE, 'Vocabulary size')

tf.flags.DEFINE_integer('vocab_max_seq_length', 10, 'Truncate or padding sequence to this length when creating vocabulary')


#Model Parameter
tf.flags.DEFINE_integer('embedding_dim', 200, 'Dimentionality of embeddings')
tf.flags.DEFINE_integer('max_num_utterance', 6, 'maximum number of utterance is a context')
tf.flags.DEFINE_integer('max_len', 6, 'maximum length of a utterance or response')
#tf.flags.DEFINE_integer('max_response_len', 50, 'maximum length of a response')
tf.flags.DEFINE_integer('negative_samples', 1, 'the negative samples in response')
tf.flags.DEFINE_integer('rnn_units', 50, 'the number of units of the hidden layer')
tf.flags.DEFINE_integer('cnn_num_filters', 8, 'the number of filters in CNN network')
tf.flags.DEFINE_string('filter_size', '3', 'the filter kernel size of each filter 3x3')
tf.flags.DEFINE_string('pooling_size', '3', 'the pooling kernel size of each stride 3x3')
tf.flags.DEFINE_integer('matching_vec_size', 30, 'the size of the final CNN output dense')


#Pre-trained Embedding
tf.flags.DEFINE_string('vocab_path', './data/vocabulary.txt', 'path to original vocabulary .txt file')
tf.flags.DEFINE_string('word2vec_path', './data/w2v_win1_d200.model', 'path to word2vec pickle model file')




#Trained Parameter
tf.flags.DEFINE_integer('batch_size', 256, 'batch size during training')
tf.flags.DEFINE_integer('eval_batch_size', 512, 'batch size during evaluation training')
tf.flags.DEFINE_integer('max_steps', 500000, 'Maximun global steps this training with run ')
tf.flags.DEFINE_float('keep_prob', 0.2, 'keep probability when adopting dropout')
tf.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate ')
tf.flags.DEFINE_string('optimizer', 'Adam', 'Optimizer name while learning')


FLAGS = tf.flags.FLAGS

HParams = namedtuple(
    'HParams',
    [
        'vocab_size',
        'embedding_dim',
        'max_num_utterance',
        'max_len',
        #'max_response_len',
        'negative_samples',
        'rnn_units',
        'cnn_num_filters',
        'filter_size',
        'pooling_size',
        'matching_vec_size',
        'vocab_path',
        'word2vec_path',
        'batch_size',
        'eval_batch_size',
        'max_steps',
        'keep_prob',
        'learning_rate',
        'optimizer',
    ])


def create_hparams():
    return HParams(
        vocab_size = FLAGS.vocab_size,
        embedding_dim = FLAGS.embedding_dim,
        max_num_utterance = FLAGS.max_num_utterance,
        max_len = FLAGS.max_len,
        #max_response_len = FLAGS.max_response_len,
        negative_samples = FLAGS.negative_samples,
        rnn_units = FLAGS.rnn_units,
        cnn_num_filters = FLAGS.cnn_num_filters,
        filter_size = map(int, FLAGS.filter_size),
        pooling_size = map(int, FLAGS.pooling_size),
        matching_vec_size = FLAGS.matching_vec_size,
        vocab_path = FLAGS.vocab_path,
        word2vec_path = FLAGS.word2vec_path,
        batch_size = FLAGS.batch_size,
        eval_batch_size = FLAGS.eval_batch_size,
        max_steps = FLAGS.max_steps,
        keep_prob = FLAGS.keep_prob,
        learning_rate = FLAGS.learning_rate,
        optimizer = FLAGS.optimizer)





