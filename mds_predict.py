#!/bin/env python
#-*- encoding:utf-8 -*-

import time
import datetime as dt
import numpy as np
import pandas as pd
import tensorflow as tf
import codecs
import mds_model
import mds_inputs
import mds_metrics
import mds_hparams
import sys
import os
 
reload(sys)
sys.setdefaultencoding('utf-8')



tf.logging.set_verbosity(tf.logging.DEBUG)

tf.flags.DEFINE_string('model_dir', None, 'Directory to load model checkpoints from')
tf.flags.DEFINE_string('vocab_processor_file', './data/vocab_processor.bin', 'Saved vocabulary processor file')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def tokenizer_fn(iterator):
    return (x.split('|') for x in iterator)


#Load test data here
#INPUT_SEQ = test_case.INPUT_SEQ1
#POTENTIAL_RESPONSE = test_case.POTENTIAL_RESPONSES


def _get_features():
    seq1 = INPUT_SEQ1
    seq1_vec = np.array(list(vp.transform([seq1.encode('utf-8')]))[0])
    seq1_len = len(seq1.split('|'))
    features = {
        'seq1': [],
        'seq1_len': [],
        'seq2': [],
        'seq2_len': []
    }
    for seq2 in POTENTIAL_RESPONSE:
        seq2_vec = np.arrary(list(vp.transform([seq2.encode('utf-8')]))[0])
        seq2_len = len(seq2.split('|'))
        features['seq1'].append(tf.convert_to_tensor(seq1_vec, dtype=tf.int64))
        features['seq1_len'].append(tf.constant(seq1_len, shape=[1,1], dtype=tf.int64))
        features['seq2'].append(tf.convert_to_tensor(seq2_vec, dtype=tf.int64))
        features['seq2_len'].append(tf.constant(seq2_len, shape=[1,1], dtype = tf.int64))

    return features



def _export_saved_model(estimator):
    #export the model for serving
    s_export = time.time()
    export_version = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    export_version = ''
    export_path = './runs/tfserving.dump/%s' % (export_version)
    print('export model begin, export_path=%s' % (export_path))
    _features, _ = _get_features()
    input_features = {}
    for k, v in _features.items():
        input_features[k] = tf.Variable(v, name = k)
    serving_input_fn = tf.contrib.learn.utils.build_default_serving_input_fn(
        features = input_features,
        default_batch_size = None)
                                
    estimator.export_savedmodel(
        export_dir_base = export_path,
        serving_input_fn = serving_input_fn,
        as_text = False)
    print('export success, dump_time=%sms'
        % (int((time.time() - s_export) * 1000)))



if __name__=='__main__':
    if not FLAG.model_dir:
        print('You must specify the dictionary, wmd features disabled default')
        print('Usage: python ./mds_predict.py --use_wmd=False, --model_dir=xxx')
        sys.exit(1)

    #Load vocabulary
    vb = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
        FLAGS.vocab_processor_file)

    hparams = mds_hparams.create_hparams()
    model_fn = mds_model.reate_model_fn(hparams)

    estimator = tf.contrib.Estimator(
        model_fn = model_fn,
        model_dir = FLAGS.model_dir)

    s = time.time()
    preds = estimator.predict(input_fn = lambda: _get_features())
    pred_res = zip(POTENTIAL_RESPONSE, preds)
    sorted_pred_res = sorted(
        pred_res, key = lambda x: x[1]['probabilities'][1], reverse = True)
    print('Context: {}\n'.format(INPUT_SEQ1.encode('utf-8')))
    for r, p in sorted_pred_res[0 : 15]:
        print 'seq2=%s' % (r.encode('utf-8'))
        print 'class=%s, prob=%0.7f\n' % (
            p['classes'], p['probabilities'][p['classes']])
    print('run_time=%sms' % (int((time.time() - s) * 1000)))

    #_export_saved_model(estimator)









