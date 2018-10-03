#! /bin/env python
#-*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
from gensim import models
import itertools
import os 
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, base_dir)


import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell as _GRUCell
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils as tf_utils
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from models import embedding_helper
import mds_hparams


MAX_SENTENCE_LEN = mds_hparams.FLAGS.max_len
MAX_NUM_UTTER = mds_hparams.FLAGS.max_num_utterance

GRU1_hidden_size = mds_hparams.FLAGS.rnn_units


kernel_size = (3, 3)
pooling_size = (3, 3)

#GRU1 = _GRUCell(num_units=mds_hparams.FLAGS.rnn_units)
#GRU2 = _GRUCell(num_units=mds_hparams.FLAGS.rnn_units)


def get_embeddings(hparams, mode, reuse = None):
    if mode == learn.ModeKeys.INFER:
        initializer = None
    elif hparams.word2vec_path and hparams.vocab_path:
        tf.logging.info("Loading word2vec embedding...")
        word2id_dict = embedding_helper.load_vocab(hparams.vocab_path)
        w2v_dict = models.word2vec.Word2Vec.load(hparams.word2vec_path)
        #w2v_dict = embedding_helper.load_w2v_vectors(hparams.word2vec_path)
        initializer = embedding_helper.build_initial_embedding_matrix(word2id_dict, w2v_dict, hparams.embedding_dim)
    else:
        tf.logging.info("pre-trained word2vec not exist, initial with random embeddings")
        initializer = np.random_uniform_initializer(-0.25, 0.25)
    with tf.variable_scope('embedding', reuse = reuse), tf.device('/cpu: 0'):
        if mode == learn.ModeKeys.INFER:
            #create a new initializer
            embeddings = tf.get_variable('word_embeddings', shape=[hparams.vocab_size, hparams.embedding_dim])
        elif hparams.word2vec_path and hparams.vocab_path:
            embeddings = tf.get_variable('word_embeddings', initializer = initializer)
        else:
            embeddings = tf.get_variable('word_embeddings', shape=[hparams.vocab_size, hparams.embedding_dim], initializer = initializer)
    return embeddings




def smn_model_fn(hparams, mode, utters, utters_len, res, res_len, input_labels):
    '''
    Sequence Matching Network: A Nueral Network for Multi-Turn Human-Computer 
    Dialogue System
    Paper:Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots
    '''
    stride_mul = lambda U, W, transpose_a = False, transpose_b = False:\
        tf.map_fn(lambda x: tf.matmul(x, W, transpose_a = transpose_a, transpose_b = transpose_b), U, dtype = tf.float32)
    
    #print("utters:", utters.shape)
    #print("utters_len:", utters_len.shape)
    #print("res:", res.shape)
    #print("res_len:", res_len.shape)
    #print("labels:", input_labels.shape)
    

    #initialize the embeddings
    embedding_W = get_embeddings(hparams, mode, reuse = None)
    
    #print("embedding_W:", embedding_W.shape)

    #U_input @shape =(max_turn, max_length)
    #U_length @shape = (max_turn, 1)
    #U_input = tf.reshape(utters, (-1,MAX_NUM_UTTER, MAX_SENTENCE_LEN))
    #U_length = tf.reshape(utters_len, (-1,MAX_NUM_UTTER))
    U_input = tf.reshape(utters, shape=[tf.shape(utters)[0], MAX_NUM_UTTER, MAX_SENTENCE_LEN])
    U_length = tf.reshape(utters_len, shape=[tf.shape(utters_len)[0], MAX_NUM_UTTER])
    #R_input = tf.reshape(res,  [MAX_SENTENCE_LEN])
    #R_length = tf.reshape(res_len, [1])
    R_input = res
    R_length = res_len

    #print("U_input:", U_input.shape)
    #print("U_length:", U_length.shape)
    #print("R_input:", R_input.shape)
    #print("R_length:", R_length.shape)
   
    # encode U and R
    # U_embedded @shape = (max_turn, max_length, embed_size)
    # R_embedded @shape = (max_length, embed_size)
    utters_emb = tf.nn.embedding_lookup(
        embedding_W, U_input, name = 'utters_emb')
    res_emb = tf.nn.embedding_lookup(
        embedding_W, R_input, name = 'res_emb')
    
    #print("utters_emb:", utters_emb.shape)
    #print("res_emb:", res_emb.shape)
    
    res_embedding = tf.transpose(res_emb, perm=[0, 2, 1])
    
    all_utterance_embeddings = tf.unstack(utters_emb, num=MAX_NUM_UTTER, axis=1)
    all_utterance_len = tf.unstack(U_length, num=MAX_NUM_UTTER, axis=1)
    #M1 = U_input * R_input'
    #@shape = (max_turn, max_length, max_length)
    #Matrix1 = stride_mul(utters_emb, res_emb, transpose_b = True)
    #Matrix1 = tf.matmul(utters_emb, res_emb)
    #print("Matrix1:", Matrix1.shape)
    #res_emb = tf.expand_dims(res_emb, axis=1)
    #print("res_emb:", res_embedding.shape)
    #U_length = tf.squeeze(U_length)
    #print("U_length:", U_length.shape)

    reuse = None
    sentence_GRU = tf.nn.rnn_cell.GRUCell(mds_hparams.FLAGS.rnn_units, kernel_initializer=tf.orthogonal_initializer())
    final_GRU = tf.nn.rnn_cell.GRUCell(mds_hparams.FLAGS.matching_vec_size, kernel_initializer=tf.orthogonal_initializer())
    # pass utterance and response through GRU1 as in the paper(see figure1)
    # @shape = (max_turn, max_length, GRU1_hidden_size)
    R_GRU1_outputs, R_GRU1_final = tf.nn.dynamic_rnn(sentence_GRU,
                                                     inputs = res_emb,
                                                     sequence_length = R_length,
                                                     dtype = tf.float32,
                                                     scope = 'sentence_GRU')
    R_GRU1_emb = tf.transpose(R_GRU1_outputs, perm=[0,2,1])
    #print('R_GRU1_emb', R_GRU1_emb.shape)
    matching_vec = []
    A_matrix = tf.get_variable('A_matrix_v', shape=(GRU1_hidden_size, GRU1_hidden_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    for utter_emb, utter_len in zip(all_utterance_embeddings, all_utterance_len):
        Matrix1 = tf.matmul(utter_emb, res_embedding)
        #print('Matrix1', Matrix1.shape)
        U_GRU1_outputs, _ = tf.nn.dynamic_rnn(sentence_GRU,
                                              inputs = utter_emb,
                                              sequence_length = utter_len,
                                              dtype = tf.float32,
                                              scope = 'sentence_GRU')
        Matrix_tmp = tf.einsum('aij,jk->aik', U_GRU1_outputs, A_matrix)
        Matrix2 = tf.matmul(Matrix_tmp, R_GRU1_emb)
        #print('Matrix2', Matrix2.shape)
        matrix = tf.stack([Matrix1, Matrix2], axis = 3)
        #print('matrix', matrix.shape)
        conv_out = tf.layers.conv2d(matrix,
                                    filters = hparams.cnn_num_filters,
                                    kernel_size = kernel_size,
                                    padding = 'VALID',
                                    kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                    activation = tf.nn.relu,
                                    reuse = reuse,
                                    name = 'conv')
        #print('conv_out',conv_out.shape)
        pooling_out = tf.layers.max_pooling2d(inputs = conv_out,
                                              pool_size = pooling_size,
                                              strides = pooling_size,
                                              padding = 'VALID',
                                              name = 'max_pooling')
        
        
        #print('pooling_out', pooling_out.shape)
        matching_vector = tf.layers.dense(inputs=tf.contrib.layers.flatten(pooling_out),
                                          units=hparams.matching_vec_size,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          activation=tf.nn.relu,
                                          reuse = reuse,
                                          name = 'matching_v')
        #print('matching_vector', matching_vector.shape)
        if not reuse:
            reuse = True
        matching_vec.append(matching_vector)
    #print('matching_vec',tf.stack(matching_vec, axis = 1).shape)
        
    #tf.get_variable_scope().reuse_variables()
        #U_GRU1_outputs, U_GRU1_final = dynamic_rnn(cell = _GRUCell(num_units=mds_hparams.FLAGS.rnn_units),
        #                                          inputs = utters_emb,
        #                                          sequence_length = U_length,
        #                                          dtype= tf.float32)

    # M2 = U_GRU1_outputs * R_GRU1_outputs'
    # @shape = (max_turn, max_length, max_length)
    #R_GRU1_outputs = tf.squeeze(R_GRU1_outputs)
    #bilinear_weight = tf.Variable(np.random.rand(GRU1_hidden_size, GRU1_hidden_size), dtype=tf.float32)
    #Matrix2 = stride_mul(U_GRU1_outputs, tf.matmul(bilinear_weight, R_GRU1_outputs, transpose_b=True))
    
    #print("Matrix2:", Matrix2.shape)
    #concatnate the Matrix1 and Matrix2 into an image of 2 channels
    #concat_out = tf.stack([Matrix1, Matrix2], axis=3)
    

    #pooling followed by CNN 
    # @shape = (max_turn, (max_length - 2)/kernel_size[0], (max_length - 2)/kernel_size[0], num_filters)
    #conv_out = tf.layers.conv2d(inputs = concat_out,
    #                            filters = hparams.cnn_num_filters,
    #                            kernel_size = kernel_size,
    #                            activation = tf.nn.relu)
    #pooling_out = tf.layers.max_pooling2d(inputs = conv_out,
    #                                      pool_size = pooling_size,
    #                                      strides = pooling_size)
    # flatten and pass through a dense layer to produce the matching vector
    # @shape = (max_turn, matching_vec_size)
    #dense_input_size = hparams.cnn_num_filters * (int((MAX_SENTENCE_LEN - 2) / kernel_size[0]) ** 2)
    #matching_vec = tf.layers.dense(inputs=tf.reshape(pooling_out, [-1, dense_input_size]),
    #                               units=hparams.matching_vec_size,
    #                               activation=tf.nn.relu)

    #print("matching_vec:", matching_vec.shape)

    #matching_vec_stack = tf.expand_dims(matching_vec, axis=0)
    #print("matching_vec_stack:", matching_vec_stack)
    #Matching Accumulation
    #@shape = (GRU2_hidden_size)

    matching_accumulation_GRU2,last_hidden = tf.nn.dynamic_rnn(final_GRU,
                                                    inputs = tf.stack(matching_vec, axis=1, name='matching_stack'),
                                                    #sequence_length = hparams.max_num_utterance,
                                                    dtype =  tf.float32,
                                                    scope = 'final_GRU')

    # logits prediction
    # @shape = ( 2)
    with tf.variable_scope('logits'):
        y = tf.layers.dense(inputs=last_hidden, units=2, activation=None, name='logits')
    
    loss = None
    train_op = None
    #print('y_shape', y.shape)
    # calculate loss (for both TRAIN and EVAL mode)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(
            indices = tf.cast(input_labels, tf.int32), depth = 2)
        #print('onehot_labels:', onehot_labels.shape)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels = onehot_labels, logits = y)



    #optimizer = tf.train.AdamOptimizer(learing_rate = hparams.learing_rate)
    #train_step = optimizer.minimize(loss=cross_entropy)
    #print(optimizer)`:
    #init = tf.global_variables_initializer()
    #saver = tf.train.Saver(max_to_keep = 15, keep_checkpoint_every_n_hours = 3)

    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss = loss,
            global_step = tf.contrib.framework.get_global_step(),
            learning_rate = hparams.learning_rate,
            optimizer = hparams.optimizer,
            clip_gradients = 10.0 )

    # construct predictions according to predefined format
    preds = {
        'classes': tf.argmax(y, axis = 1),
        'probabilities': tf.nn.softmax(y, name = 'softmax_tensor')#[:,1]
    }

    # return a ModelFnOps object
    return model_fn_lib.ModelFnOps(mode = mode, predictions = preds,
                                   loss = loss, train_op = train_op)


