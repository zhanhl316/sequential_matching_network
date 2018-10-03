#!/bin/env python
#-*- encoding:utf-8 -*-


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib import learn

import os
import io
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, base_dir)

from models import SMN_model
import mds_hparams


def get_id_features(features, key, len_key, max_len):
    idx = features[key]
    #Note: slice batched sequences to their max_len if necessary
    idx = tf.slice(idx, begin = [0,0], size = [-1, max_len])
    idx_len = tf.squeeze(features[len_key], [1])
    idx_len = tf.minimum(idx_len, tf.constant(max_len, dtype=tf.int64))
    return idx, idx_len


def get_utters_id_features(features, key, len_key, max_len, max_utter):
    idx = features[key]
    #Note: slice batched sequences to their max_len if necessary
    idx = tf.slice(idx, begin = [0,0], size = [-1, max_len])
    idx_len = tf.slice(features[len_key], begin = [0,0],size = [-1, max_utter])
    #idx_len = tf.squeeze(features[len_key], [6])
    #idx_len = tf.minimum(idx_len, tf.constant(max_len, dtype=tf.int64))
    return idx, idx_len


#Note: the following wrapper MUST be provided, because the Estimators only recognize
# the signature with format of (features, labels, mode)
def create_model_fn(hparams):
    def model_fn(features, label_dict, mode):
        utters, utters_len = get_utters_id_features(features, 'utters', 'utters_len', hparams.max_len*hparams.max_num_utterance, hparams.max_num_utterance)
        #print('aaaaa, utters_shape', utters.shape)
        #print('aaaaa, utters_len', utters_len.shape)
        res, res_len = get_id_features(features, 'res', 'res_len', hparams.max_len)
        #extra_features = features['extra_features']
        #print('aaaaa, res_shape', res.shape)
        #print('aaaaa, res_len', res_len.shape)


        if mode != learn.ModeKeys.INFER:
            labels = label_dict['labels']
            #print('aaaaa,labels', labels.shape)
            labels = tf.squeeze(labels, [1])
            #print('aaaaa, labels', labels.shape)

        if mode == learn.ModeKeys.TRAIN:
            model_fn_ops = SMN_model.smn_model_fn(
                hparams,
                mode,
                utters,
                utters_len,
                res,
                res_len,
                input_labels = labels)
            return model_fn_ops

        elif mode == learn.ModeKeys.INFER:
            model_fn_ops = SMN_model.smn_model_fn(
                hparams,
                mode,
                utters,
                utters_len,
                res,
                res_len,
                input_labels = None)
            return model_fn_ops

        elif mode == learn.ModeKeys.EVAL:
            all_seq1s = [utters]
            all_seq1_lens = [utters_len]
            all_seq2s = [res]
            all_seq2_lens = [res_len]
            
            distractor_cnt = 9
            for i in range(distractor_cnt):
                distractor, distractor_len = get_id_features(
                    features,
                    'distractor_{}'.format(i),
                    'distractor_{}_len'.format(i),
                    hparams.max_len)
                all_seq1s.append(utters)
                all_seq1_lens.append(utters_len)
                all_seq2s.append(distractor)
                all_seq2_lens.append(distractor_len)
            
            # use tf.concat to adjust batch_size (new_size = 10 * batch_size)
            model_fn_ops = SMN_model.smn_model_fn(
                hparams,
                mode,
                tf.concat(values = all_seq1s, axis = 0),
                tf.concat(values = all_seq1_lens, axis = 0),
                tf.concat(values = all_seq2s, axis = 0),
                tf.concat(values = all_seq2_lens, axis = 0),
                input_labels = labels)
            
            #NOTE: split to make sure the groud truth and its distractors
            # is in the same group
            #print('probabolities:',model_fn_ops.predictions['probabilities'])
            split_probs = tf.split(
                value = model_fn_ops.predictions['probabilities'],
                num_or_size_splits = distractor_cnt + 1,
                axis = 0)
            #print('split_probs:', split_probs)
            shaped_probs = tf.concat(split_probs, axis = 1)
            ## use tf.map_fn to extract prob of the ground truth label (lable_idx=1)
            sliced_probs = tf.map_fn(lambda x: x[1::2], shaped_probs)
            preds = {
                'classes': model_fn_ops.predictions['classes'],
                'probabilities': sliced_probs,
            }
            return model_fn_lib.ModelFnOps(
                mode = mode, predictions = preds,
                loss = model_fn_ops.loss, train_op = None)
    
    return model_fn
