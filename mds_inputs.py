#!/bin/env python
#-*- encoding:utf-8 -*-


import numpy as np
import codecs
import sys
import os
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
import mds_hparams

TEXT_FEATURE_SIZE = mds_hparams.FLAGS.max_len
TURN_FEATURE_SIZE = mds_hparams.FLAGS.max_num_utterance
#EXTRA_FEATURE_SIZE = 3
DISTRACTOR_COUNT = 9



def get_feature_column(mode):
    feature_columns = []

    feature_columns.append(layers.real_valued_column(
        column_name = 'res', dimension = TEXT_FEATURE_SIZE, dtype = tf.int64))
    feature_columns.append(layers.real_valued_column(
        column_name = 'res_len', dimension = 1, dtype = tf.int64))

 
    feature_columns.append(layers.real_valued_column(
        column_name = 'utters', dimension = TEXT_FEATURE_SIZE*TURN_FEATURE_SIZE, dtype = tf.int64))
    feature_columns.append(layers.real_valued_column(
            column_name = 'utters_len', dimension = TURN_FEATURE_SIZE, dtype = tf.int64))

    if mode == learn.ModeKeys.TRAIN:
        feature_columns.append(layers.real_valued_column(
            column_name = 'label', dimension = 1, dtype = tf.int64))
    elif mode == learn.ModeKeys.EVAL:
        for i in xrange(DISTRACTOR_COUNT):  
            feature_columns.append(layers.real_valued_column(
                column_name = 'distractor_{}'.format(i),
                dimension = TEXT_FEATURE_SIZE,
                dtype = tf.int64))
            feature_columns.append(layers.real_valued_column(
                column_name = 'distractor_{}_len'.format(i),
                dimension = 1,
                dtype = tf.int64))

    #print('feature_columns=%s' % (feature_columns))
    return set(feature_columns)



def create_input_fn(mode, input_files, batch_size, num_epochs):
    def input_fn():
        features = layers.create_feature_spec_for_parsing(
            get_feature_column(mode))

        feature_map = learn.read_batch_features(
            file_pattern = input_files,
            batch_size = batch_size,
            features = features,
            reader = tf.TFRecordReader,
            randomize_input = True,
            num_epochs = num_epochs,
            queue_capacity = 200000 + batch_size * 5,
            name = 'read_batch_features_{}'.format(mode))

        if mode == learn.ModeKeys.TRAIN:
            target = feature_map.pop('label')
            labels = {'labels':target}
        else:
            # NOTE: sample count of the last batch maybe less than batch_size
            # thus batch_size should be adjusted to the exact value here
            exact_batch_shape = feature_map['res_len']
            # construct labels for recall/precision metrics
            recall_target = tf.zeros_like(exact_batch_shape, dtype = tf.int64)
            # construct labels for accuracy metrics
            accuracy_target = [tf.ones_like(exact_batch_shape, dtype = tf.int64)]
            for i in xrange(DISTRACTOR_COUNT):
                accuracy_target.append(
                    tf.zeros_like(exact_batch_shape, dtype = tf.int64))
            accuracy_target = tf.concat(values = accuracy_target, axis = 0)
            labels = {'labels': accuracy_target, 'recall_labels': recall_target}
        
        #print('feature_map=%s' % (feature_map))
        #print('labels=%s' % (labels))
        return feature_map, labels
    
    return input_fn


