#!/bin/env python
#-*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import io
import os
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, base_dir)

from itertools import chain
from tensorflow import expand_dims
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sets
from tensorflow.python.ops import nn
import nlp_util_py3
import conf
import mds_hparams
import embedding_helpers


def recall_at_k(labels, predictions, k):
    '''
    Compute recall at position k.

    :param labels: shape=(num_examples,), dtype=tf.int64
    :param predictions: logits of shape=(num_examples, num_classes)
    :param k: recall position
    :return: recall at position k


    Example:

    labels = tf.constant([0, 1, 1], dtype=tf.int64)
    predictions = tf.constant([[0.1, 0.2, 0.3], [3, 5, 2], [0.3, 0.4, 0.7]])
    recall_at_k(labels, predictions, 2)
    # recall_at_k(labels, predictions, 2) = 0.6667

    '''
    labels = expand_dims(labels, axis=1)
    _, predictions_idx = nn.top_k(predictions, k)
    predictions_idx = math_ops.to_int64(predictions_idx)
    tp = sets.set_size(sets.set_intersection(predictions_idx, labels))
    tp = math_ops.to_double(tp)
    tp = math_ops.reduce_sum(tp)
    fn = sets.set_size(sets.set_difference(predictions_idx, labels, aminusb=False))
    fn = math_ops.to_double(fn)
    fn = math_ops.reduce_sum(fn)
    recall = math_ops.div(tp, math_ops.add(tp, fn), name='recall_at_k')

    return recall


def _word2idx_mapping(sentence, word2id_dict):
    idx_list = [
        word2id_dict.get(w, word2id_dict['<UNK>']) for w in sentence]
    return idx_list


def _pad_sent(sentence, max_l):
    """
    pad sentence with 0's, or throw away words from the tail of sentence
    if its length is longer than `max_l`
    """
    sent_length = len(sentence)
    sentence = (sentence[:max_l] if sent_length > max_l
        else sentence + [0] * (max_l - sent_length))
    if sent_length > max_l:
        sent_length = max_l
    sentence.append(sent_length)
    return sentence

def len_sent(sentence):
    cnt = 0
    line = sentence.strip('\n').split(' ')
    for word in line:
        if word != '':
            cnt += 1
    return cnt

def padding_sent_orig(sentence, max_len):
    '''
    pad sentence with <UNK>'s, or throw away words from the tail os sentence 
    if its length is longer than 'max_len'
    '''

    sent_length = len_sent(sentence)
    if sent_length <= max_len:
        sentence = sentence + ' UNK'*(max_len - sent_length)
    else:
        sentence = ' '.join(sentence.split(' ')[:max_len])
    #sentence = (sentence[:max_len] if sent_length > max_len
    #            else sentence + ' UNK'* (max_len - sent_length) )
    return sentence


def padding_len_crop(file_from, file_to, max_len):
    '''
    padding session with the same max_len * max_utter
    '''
    querys = []
    respos = []

    with io.open(file_from, 'r', encoding='utf-8') as fp_from, io.open(file_to, 'w', encoding='utf-8') as fp_to:
        dialogue = []
        for line in fp_from:
            if line.strip('\n') == '':
                if len(dialogue) > 0:
                    querys.append(dialogue[0:-1])
                    respos.append(dialogue[-1] + '\n')
                dialogue = []
                #querys_len = len_sent(querys)
                continue
            line = padding_sent_orig(line.strip('\n'), max_len)
            dialogue.append(line + '\n' )

        for idx in range(10000):#remember to change the parameter
        # write the true context-reponse pair
            fp_to.write(unicode(''.join(querys[idx])))
            fp_to.write(unicode(respos[idx]))

     
def padding_utter_crop(file_from, file_to, max_len, max_utter):
    '''
    padding session with the same max_len*max_utter
    '''
    cnt = 0
    querys = []
    respos = []
    with io.open(file_from, 'r', encoding='utf-8') as fp_from, io.open(file_to, 'w', encoding='utf-8') as fp_to:
        dialogue = []
        for line in fp_from:
            if line !='\n':
                cnt += 1
                dialogue.append(line)
            if line == '\n':
                if len(dialogue) > 0:
                    if cnt <= max_utter:
                        querys.append(dialogue[0:-1])
                        fp_to.write(unicode(''.join(querys[0])))
                        for i in range(max_utter - cnt + 1):
                            unk_sent = u'UNK '*max_len
                            fp_to.write(unicode(' '.join(unk_sent.split(' ')[:max_len]) + '\n'))
                    else:
                        querys.append(dialogue[0:max_utter])
                        fp_to.write(unicode(''.join(querys[0])))
                    respos.append(dialogue[-1] + '\n')
                    fp_to.write(unicode(respos[0]))
                querys = []
                respos = []
                dialogue = []
                cnt = 0
                continue



def _crop_context(context, max_turn):
    return context if len(context) <= max_turn else context[-max_turn:]


def generate_batch(fp, batch_size, word_to_idx, max_l, max_turn, neg_per_mix):
    '''
    generate a batch of data from file pointer

    returns:
        data @shape = (batch_size, (max_turn + 1)*(max_length + 1))
        labels @shape = (batch_size,)
        session_masks @shape = (batch_size, max_turn)
    '''
    # read one batch of sessions from file pointer
    tknz_func = nlp_util_py3.NLPUtil.tokenize_via_jieba
    batch_data = []
    for i in range(batch_size):
        tmp = []
        session = {}
        while True:
            line = fp.readline()
            # wrap over when read to the end of file
            if line == '':
                fp.seek(0)
                line = fp.readline()
            if line == '\n':
                session['r'] = tmp[-1]
                session['c'] = _crop_context(tmp[:-1], max_turn)
                session['l'] = int(i % (neg_per_mix+1) == 0)
                break
            _idx_sent = _word2idx_mapping(tknz_func(line.strip()), word_to_idx)
            tmp.append(_pad_sent(_idx_sent, max_l))
        batch_data.append(session)

    # resolve padding
    labels = []
    data = []
    session_masks = []
    pad_sent = [0 for _ in range(max_l+1)]
    for session in batch_data:
        labels.append(session['l'])
        response = session['r']
        session_input = session['c']
        num_utterances = len(session_input)
        session_masks.append([1 for _ in range(num_utterances)] + [0 for _ in range(max_turn - num_utterances)])
        session_input = session_input + [pad_sent for _ in range(max_turn - num_utterances)]
        session_input.append(response)
        session_input = list(chain.from_iterable(session_input))
        data.append(session_input)

    # generate random initialized embedding for new words
    return data, labels, session_masks


if __name__ == '__main__':
   
    sent = u'我们 爱 中国 非常 爱 非常 你 说 对不 对呀 是的 啊'
    
    print ("the lenth:",str(len_sent(sent)))
    
    sentence = padding_sent_orig(sent, 10)

    sentence = sentence + '|'
    print sentence.encode('utf-8')
    
    msgs = [
        u'携带乙肝病毒可以母乳喂养吗',
        u'做糖筛是不是又要打B超哦',
        u'这个crp偏高是怎么回事, 12mg, 12ml, 12mml, 11kg, 11kcal, 11k, 11kj',
        u'b 你好 乳头内陷要怎么母乳',
    ]
    #msg_len = len(msgs)
    #print str(msg_len)
    #x = []
    #for msg in msgs:
    #    x.append( padding_sent_orig(msg, 16))
    #    print x
    #for i in range(5):
    #    x.append('UNK'*30)

    #for a in x:
    #    print a.encode('utf-8')

    padding_len_crop('./data/task5_fenci.txt', './data/task5_unk.txt', mds_hparams.FLAGS.max_len)
    padding_utter_crop('./data/task5_unk.txt', './data/task5_pad.txt', mds_hparams.FLAGS.max_len, mds_hparams.FLAGS.max_num_utterance)
    #vocab_fpath = './data/vocabulary.txt'
    #word2id_dict = embedding_helpers.load_vocab(vocab_fpath)
    #for word in word2id_dict:
    #    writer_iter = word2id_dict[word]
    #    print writer_iter

    #train_fpath = './data/train_1000.txt'
    #with io.open(train_fpath, 'r', encoding='utf-8') as fp:
    #    batch_size = mds_hparams.FLAGS.batch_size
    #    max_len =  mds_hparams.FLAGS.max_len
    #    max_turn = mds_hparams.FLAGS.max_num_utterance
    #    neg_per_mix = 1
    #    #neg_per_mix = 9 # for valid or test
    #    data = generate_batch(fp, batch_size, word2id_dict, max_len, max_turn, neg_per_mix)
    #    for idx, content in enumerate(data):
    #        print str(content[idx])
