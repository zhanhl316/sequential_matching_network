#!/bin/env python
#-*- encoding:utf-8 -*-


import numpy as np
import pandas as pd
import os

g_ud_words_cfg = []

g_synonym_cfg = []

g_stop_words_cfg = set(['%%', u'？', '?', ':', '{', '}', '_', '\n', '-', ' ', ',', u'，',
    u'〞', '^', '.', ';', u'；', u'、', '!', u'！', u'：', u'（', u'）', u'(',
    u')', u'～', '∼', '"', u'“', u'”', u':)', u'/::)', u'…', '[', ']', u'【',
    u'︰', u'﹉', u'﹏', u'﹐', u'﹑', u'﹒', u'﹔', u'﹕', u'﹙', u'﹚', u'＆',
    u'＇', u'＂', u'﹎', u'o_o', u'〗', u'', u'', u'&amp', u'&gt', u'&lt',
    u'】', u'。', u'请', u'请问', u'了',
     '|',
])


