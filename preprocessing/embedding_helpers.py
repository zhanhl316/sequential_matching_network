#! /bin/env python
#-*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import codecs
import jieba
import collections
from gensim import models
import time
import os
import io
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, base_dir)

import mds_hparams
import conf


def load_vocab(vocab_fpath):
    vocab = None
    with codecs.open(vocab_fpath, 'r', 'utf-8') as rfd:
        vocab = rfd.read().splitlines()
    word2id_dict = {}
    for idx, word in enumerate(vocab):
        word2id_dict[word] = idx
    tf.logging.info("load vocab successfully, vocab_size = %s" % (len(word2id_dict)))
    return word2id_dict
    


def load_w2v_vectors(w2v_fpath):
    w2v_dict = {}
    with codecs.open(w2v_fpath, 'r', 'utf-8') as rfd:
        for line in rfd:
            fields = line.strip('\n').split(' ')
            word = fields[0]
            idx = map(float, fields[1:])
            w2v_dict[word] = np.array(idx, dtype=np.float32)
    tf.logging.info("load w2v model successfully, w2v_size = %s" % (len(w2v_dict)))
    return w2v_dict
    


def build_initial_embedding_matrix(word2id_dict, w2c_dict, embedding_dim):
    initial_embeddings = np.random.uniform(0.25, -0.25, (len(word2id_dict), embedding_dim)).astype(np.float32)
    for word, idx in word2id_dict.items():
        if word in w2v_dict:
            initial_embeddings[idx, :] = w2v_dict[word]
    tf.logging.info("Build intital embedding successfully")
    return initial_embeddings


def tokenize_corpus(corpus_fpath, save_fpath, is_train_data = True):
    
    def processLine(line, is_train_data = True):
        line = line.strip('\n')
        seg_list = jieba.cut(line)
        stop_word_set = conf.g_stop_words_cfg
        seg_list = filter(lambda w: w not in stop_word_set, seg_list)
        content = ' '.join(seg_list)
        return (content + '\n')

    with codecs.open(corpus_fpath, 'r', 'utf-8') as in_f, \
        codecs.open(save_fpath, 'w', 'utf-8') as out_f:
            for line in in_f.readlines():
                out_f.write(processLine(line, is_train_data))
    print 'Tokenize Done'





def getCorpus(corpus_fpath):
    corpus = []
    #fenciTotalFile = open('fencitotalContext.txt', 'r')
    #for line in fenciTotalFile:
    with codecs.open(corpus_fpath, 'r', 'utf-8') as in_f:
        for line in in_f:
            if line !='\n':
                corpus_tmp = line.strip().split('|')
                corpus.extend(corpus_tmp)
    print 'Get corpus done, length is %d' % len(corpus)
    print str(corpus[100])
    return corpus


def train_word2vec(corpus, wv_fpath):
    time_s = time.time()
    vec_size = mds_hparams.FLAGS.embedding_dim
    win_size = 1
    print ('begin to train model...')
    w2v_model = models.word2vec.Word2Vec(corpus,
                                         size = vec_size,
                                         window = win_size,
                                         min_count = 2,
                                         workers = 4,
                                         sg = 1,
                                         negative = 15,
                                         iter = 7)
    w2v_model.train(corpus, total_examples = len(corpus), epochs = w2v_model.iter)
    save_fpath = os.path.join(wv_fpath,
        'w2v_win%s_d%s.model' % (win_size, vec_size))
    w2v_model.save(save_fpath)
    print ('save model success, model_path=%s, time=%.4f sec.'
        % (save_fpath, time.time() - time_s))


def train_word2vec_model(corpus_, model_fpath, wv_fpath = None):
    vec_size = mds_hparams.FLAGS.embedding_dim
    win_size = 1
    #corpus_ = list(_input_streaming(corpus_fpath))
    w2v_model = models.word2vec.Word2Vec(corpus_,
                                         size = vec_size,
                                         window = win_size,
                                         min_count = 30,
                                         workers = 40,
                                         sg = 1,
                                         negative = 15,
                                         iter = 7)
     # begin to train
    print('begin to train model...')
    w2v_model.train(corpus_, total_examples = len(corpus_), epochs = w2v_model.iter)
    print('begin to save model')
    save_fpath = os.path.join(model_fpath,
        'w2v_win%s_d%s.model' % (win_size, vec_size))
    w2v_model.save(save_fpath)
    if wv_fpath:
        wv = w2v_model.wv
        fname = 'w2v_sgns_win%s_d%s.kv' % (win_size, vec_size)
        wv.save_word2vec_format(fname)
    print('save model success, model_path=%s'
        % (model_fpath))



if __name__ == '__main__':
    
    tokenize_corpus('./data/task5test_correct.txt', './data/task5_fenci.txt', is_train_data = True)
    #tokenize_corpus('./data/train_3000000.txt', './data/fencitotalContext.txt', is_train_data = True)

    #corpus = getCorpus('./data/fencitotalContext.txt')

    #Train word2vec
    #train_word2vec_model(corpus, './data', wv_fpath = None)
