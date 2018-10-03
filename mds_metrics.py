#!/bin/env python 
#-*- encoding:utf-8 -*-


import numpy as np
import functools
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.learn import MetricSpec



def create_evaluation_metrics():
    eval_metrics = {}
    for k in [1, 3, 5, 10]:
        recall_metric_spec = MetricSpec(
            metric_fn = functools.partial(tf.metrics.recall_at_k, k = k),
            prediction_key = 'probabilities',
            label_key = 'recall_labels')
        precision_metric_spec = MetricSpec(
            metric_fn = functools.partial(tf.metrics.sparse_precision_at_k, k = k),
            prediction_key = 'probabilities',
            label_key = 'recall_labels') 
        eval_metrics['recall_at_%d' % k] = recall_metric_spec
        eval_metrics['precision_at_%d' % k ] = precision_metric_spec
    eval_metrics['accuracy'] = MetricSpec(
        metric_fn = tf.metrics.accuracy,
        prediction_key = 'classes',
        label_key = 'labels')
    
    return eval_metrics

