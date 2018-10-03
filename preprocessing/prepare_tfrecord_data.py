#!/bin/env python 
#-*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import io
import codecs
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, base_dir)


import pickle
import csv
import tensorflow as tf
import functools
from tensorflow.contrib.learn import preprocessing
import mds_hparams
import pickle
import nlp_util_py3


MAX_SENTENCE_LEN = mds_hparams.FLAGS.max_len
MAX_NUM_UTTER = mds_hparams.FLAGS.max_num_utterance

tf.flags.DEFINE_string('input_dir', os.path.abspath('./data'), 'Input dictionary containing original .txt file')
tf.flags.DEFINE_string('output_dir', os.path.abspath('./data'), 'Output dictionary containing TFRecord file')
tf.flags.DEFINE_integer('min_word_freq', 1, 'minimum number of word frequency in the vocabulary')


FLAGS = tf.flags.FLAGS
# hack here because tf.flags.FLAG only parsed once automatically, and
# new flag defined in this file will not be contained due to parsed already
# in mds_hparams.py, thus we call it manually here
FLAGS._parse_flags()



TRAIN_PATH = os.path.join(FLAGS.input_dir, 'train_test.csv')
VALID_PATH = os.path.join(FLAGS.input_dir, 'mvalid_1000lad_comb.csv')
TEST_PATH = os.path.join(FLAGS.input_dir, 'task5_pad_comb.csv')


def create_file_iter(filename):
    with io.open(filename, 'r', encoding='utf-8') as f:
        for row in f:
            if len(row):
                yield row


def create_csv_iter(filename):
    """
        Return an iterator over a CSV file. Skips the header if necessary.
            """
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header
        #next(reader)
        for row in reader:
            yield row

def tokenizer_fn(iterator):
    tknz_func = nlp_util_py3.NLPUtil.tokenize_via_jieba
    return (tknz_func(x) for x in iterator)



def create_vocab(input_iter, min_freq):
    print('begin ---')
    vp = preprocessing.VocabularyProcessor(
        max_document_length = MAX_SENTENCE_LEN * MAX_NUM_UTTER,
        min_frequency = min_freq,
        tokenizer_fn = tokenizer_fn)
    print('begin fit -----')
    vp.fit(input_iter)
    return vp
   

def transform_sentence(sequence, vocab_processor):
    '''
    Map a sentence into the integer vocabulary. 
    '''
    return next(vocab_processor.transform([sequence])).tolist()


def create_example_train(row, vocab):
    '''
    Return a tensorflow.Example Protocol Buffer object.
    '''
    # list of utterance, response and label
    utters, res  = row[0:2]
    label = row[-1]
    utters_len = []
    utters_transformed = transform_sentence(utters, vocab)
    #print('xxxxxxxxx, debug, utters=%s' % (utters))
    #print('xxxxxxxxx, debug, utters_transformed=%s' % (utters_transformed))
    res_transformed = transform_sentence(res, vocab)[0:MAX_SENTENCE_LEN]
    
    #print('xxxxxxxxx, debug, res=%s' % (res))
    #print('xxxxxxxxx, debug, res_transformed=%s' % (res_transformed))
    #utters_len = len(next(vocab._tokenizer([utters])))
    #res_len = len(next(vocab._tokenizer([res])))
    tmp = len(utters_transformed)/MAX_SENTENCE_LEN
    for i in range(tmp):
        utters_len.append(MAX_SENTENCE_LEN)
    res_len = len(res_transformed)
    label = int(float(label))


    #print ("utters_len:",str(utters_len))
    #print ("res_len:",str(res_len))
    #print  str(label)



    #create example
    example = tf.train.Example()
    example.features.feature['utters'].int64_list.value.extend(utters_transformed)
    example.features.feature['utters_len'].int64_list.value.extend(utters_len)
    example.features.feature['res'].int64_list.value.extend(res_transformed)
    example.features.feature['res_len'].int64_list.value.extend([res_len])
    example.features.feature['label'].int64_list.value.extend([label])
    #print('Successfully built')
    #print('example=%s' % (example))
    return example




def create_example_test(row, vocab):
    '''
    Create validation/test set 
    Return a tensorflow.Example Protocol Buffer object.
    '''
    utters, res  = row[0:2]
    distractors = row[2:]
    utters_len = []

    utters_transformed = transform_sentence(utters,vocab)
    res_transformed = transform_sentence(res,vocab)[0:MAX_SENTENCE_LEN]
    #print(len(next(vocab._tokenizer([utters]))))
    #print(len(next(vocab._tokenizer([res]))))
    tmp = len(utters_transformed)/MAX_SENTENCE_LEN
    for i in range(tmp):
        utters_len.append(MAX_SENTENCE_LEN)
    res_len = len(res_transformed)


    #create example
    example = tf.train.Example()
    example.features.feature['utters'].int64_list.value.extend(utters_transformed)
    example.features.feature['utters_len'].int64_list.value.extend(utters_len)
    example.features.feature['res'].int64_list.value.extend(res_transformed)
    example.features.feature['res_len'].int64_list.value.extend([res_len])


    #print ("utters_len:",str(utters_len))
    #print ("res_len:",str(res_len))

    #add distractor
    dis_sample_cnt = len(distractors)
    #print('dis_sample_cnt', str(dis_sample_cnt))
    for i in xrange(dis_sample_cnt):
        dis_key = 'distractor_{}'.format(i)
        dis_len_key = 'distractor_{}_len'.format(i)
        dis_item = distractors[i:i+1]
        distractor = dis_item[0]

        #dis_len = len(next(vocab._tokenizer([distractor])))
        dis_len = MAX_SENTENCE_LEN
        example.features.feature[dis_len_key].int64_list.value.extend([dis_len])
        distractor_transformed = transform_sentence(distractor, vocab)[0:MAX_SENTENCE_LEN]
        dis_len = len(distractor_transformed)
        example.features.feature[dis_key].int64_list.value.extend(distractor_transformed)
    
    #print('example=%s' % (example)) 
    return example
    

def create_tfrecord_file(input_fname, output_fname, example_fn):
    '''
    Create a TFRecords file for the given input data and
    example transofmration function
    '''
    print('Creating tfRecords file at {}...'.format(output_fname))
    with tf.python_io.TFRecordWriter(output_fname) as writer:
        for i, row in enumerate(create_csv_iter(input_fname)):
            x = example_fn(row)
            writer.write(x.SerializeToString())
    print('Finish writing to {}'.format(output_fname))


def write_vocabulary(vocab_processor, outfile):
    '''
    Write the vocabulary to a file, one word one line
    '''
    vocab_size = len(vocab_processor.vocabulary_)
    vocab_list = [vocab_processor.vocabulary_._reverse_mapping[id_] 
        for id_ in xrange(vocab_size)]
    with codecs.open(outfile, 'w') as wfd:
        wfd.write('\n'.join(vocab_list))
    print('Saved vocabulary to {}'.format(outfile))



if __name__ == '__main__':
    print('Createing Vocabulary...')
    #input_fpath = './data/train_10000.txt'
    output_fdir = './data/'
    #input_iter = create_file_iter('./data/train_3000000.txt')
    #input_iter = create_csv_iter('./data/train_test.csv')
    #input_iter = (x[0] + '|' + x[1] for x in input_iter)
    #print('Create vocab...')
    #vocab = create_vocab(input_iter,min_freq = 1)
    #print('The total vocabulary size: {}'.format(len(vocab.vocabulary_)))
    
    #create vocabulary.txt file
    #write_vocabulary(vocab, os.path.join(output_fdir, 'vocabulary.txt'))
    
    #Save the vocab processor as a pickle format
    #vocab.save(os.path.join(FLAGS.output_dir, 'vocab_processor.bin'))
    vocab = preprocessing.VocabularyProcessor.restore(os.path.join(FLAGS.output_dir, 'vocab_processor.bin'))
    #Create train.tfRecords
    #create_tfrecord_file(
    #    input_fname = TRAIN_PATH,
    #    output_fname = os.path.join(FLAGS.output_dir, 'train.tfrecords'),
    #    example_fn = functools.partial(create_example_train, vocab = vocab))

    #Create valid.tfRecords
    create_tfrecord_file(
        input_fname = VALID_PATH,
        output_fname = os.path.join(FLAGS.output_dir, 'valid_1000.tfrecords'),
        example_fn = functools.partial(create_example_test, vocab = vocab))
    
    #create test.tfRecords
    create_tfrecord_file(
        input_fname = TEST_PATH,
        output_fname = os.path.join(FLAGS.output_dir, 'task5.tfrecords'),
        example_fn = functools.partial(create_example_test, vocab = vocab))
