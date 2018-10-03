#! /bin/env python
#-*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import codecs


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
    


def build_initial_embedding_matrix(word2id_dict, w2v_dict, embedding_dim):
    initial_embeddings = np.random.uniform(0.25, -0.25, (len(word2id_dict), embedding_dim)).astype(np.float32)
    for word, idx in word2id_dict.items():
        if word in w2v_dict:
            initial_embeddings[idx, :] = w2v_dict[word]
    tf.logging.info("Build intital embedding successfully")
    return initial_embeddings





