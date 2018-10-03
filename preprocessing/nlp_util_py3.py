#!/bin/env python
#-*- encoding: utf-8 -*-


import codecs
import re
from collections import Counter
import jieba
import conf
from log import g_log_inst as logger


class NLPUtil(object):
    _valid_token_len = 5

    _wordseg_pattern_cfg = [
        re.compile(r'{.*?}', re.U),
    ]

    _emoji_pattern_cfg = re.compile('[\U00010000-\U0001ffff]', re.U)

    _replace_pattern_cfg = {
        'float_t': re.compile('\d+\.\d+'),
        'phone_t': re.compile('1[0-9]{10}'),
        'email_t': re.compile('[^@|\s]+@[^@]+\.[^@|\s]+'),
    }

    _illegal_char_set = set([])

    # init jieba
    jieba.initialize()
    ud_words = conf.g_ud_words_cfg
    for w in ud_words:
        jieba.add_word(w, freq = 100000)


    @classmethod
    def remove_illegal_gbk_char(cls, text_unicode):
        try:
            text_unicode.encode('gbk')
            return text_unicode
        except UnicodeEncodeError as e:
            illegal_ch = e.object[e.start : e.end]
            illegal_set = cls._illegal_char_set
            illegal_set.add(illegal_ch)
            # try to replace directly
            for ch in illegal_set:
                text_unicode = text_unicode.replace(ch, '')
            # remove recursively
            return cls.remove_illegal_gbk_char(text_unicode)


    @classmethod
    def remove_emoji_char(cls, text_unicode):
        res = cls._emoji_pattern_cfg.sub('', text_unicode)
        return res


    @classmethod
    def conv_fenc_u8_to_gbk(cls, in_fpath, out_fpath):
        try:
            with codecs.open(in_fpath, 'r', 'utf-8') as rfd, \
                codecs.open(out_fpath, 'w', 'gbk') as wfd:
                # read utf8, write gbk
                for line in rfd:
                    line = cls.remove_illegal_gbk_char(line)
                    wfd.write(line)
        except Exception as e:
            logger.get().warn('errmsg=%s' % (e))


    @classmethod
    def tokenize_via_jieba(cls, text, filter_stop_word = True, norm_flag = True):
        tokens = jieba.lcut(text.lower())
        if filter_stop_word:
            stop_words = conf.g_stop_words_cfg
            tokens = filter(lambda x: x not in stop_words, tokens)
            if norm_flag:
                norm_func = cls._normalize_token
                return map(norm_func, tokens)
            else:
                return tokens
        else:
            return tokens

    @classmethod
    def stat_token_freq(cls, in_fpath, out_fpath):
        stop_words = conf.g_stop_words_cfg
        try:
            word_counter = Counter()
            with codecs.open(in_fpath, 'r', 'utf-8') as rfd:
                for line in rfd:
                    raw_str, word_seg = line.strip('\n').split('\t')
                    tokens = word_seg.split()
                    tokens = filter(lambda x: x not in stop_words, tokens) 
                    tokens = map(cls._normalize_token, tokens)
                    for t in tokens:
                        if ('{[' not in t) and len(t) <= cls._valid_token_len:
                            word_counter[t] += 1
                        else:
                            logger.get().warn('invalid token, token=%s' % (t))
                            # tokenize via jieba 
                            for n_t in jieba.cut(t):
                                word_counter[n_t] += 1
                                logger.get().debug('jieba cut, token=%s' % (n_t))
            # dump word_counter
            sorted_words = sorted(word_counter.keys(),
                key = lambda k: word_counter[k], reverse = True)
            with codecs.open(out_fpath, 'w', 'utf-8') as wfd:
                for word in sorted_words:
                    tmp = '%s\t%s\n' % (word, word_counter[word]) 
                    wfd.write(tmp)
        except Exception as e:
            logger.get().warn('errmsg=%s' % (e))


    @classmethod
    def _normalize_token(cls, token):
        token = token.lower()
        try:
            # 11 usually means phone number
            if len(token) != 11 and token.isdigit():
                token = 'int_t'
            for k, v in cls._replace_pattern_cfg.items():
                if v.match(token):
                    token = k
                    break
            if '{[' not in token:
                return token
            for item in cls._wordseg_pattern_cfg:
                token = item.sub('', token)
            return token
        except Exception as e:
            logger.get().warn('token=%s, errmsg=%s' % (token, e))
            return token


if '__main__' == __name__:
    logger.start('./log/test.log', __name__, 'DEBUG')

    in_fpath = './data/question.raw'
    out_fpath = './data/question.raw.gbk'
    #NLPUtil.conv_fenc_u8_to_gbk(in_fpath, out_fpath)

    
    in_fpath = './data/question.seg.u8'
    out_fpath = './data/vocab.txt'
    #NLPUtil.stat_token_freq(in_fpath, out_fpath)

    msgs = [
        u'携带乙肝病毒可以母乳喂养吗',
        u'做糖筛是不是又要打B超哦',
        u'这个crp偏高是怎么回事, 12mg, 12ml, 12mml, 11kg, 11kcal, 11k, 11kj',
        u'b 你好 乳头内陷要怎么母乳',
    ]
    for msg in msgs:
        x = NLPUtil.tokenize_via_jieba(msg)
        print x
        print '|'.join(x).encode('utf-8')
