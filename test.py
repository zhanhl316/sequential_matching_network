#!bin/env python
#-*- coding: utf-8 -*-


import tensorflow as tf 
import numpy as np
import pandas as pd


a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.reshape(a, (1,6))

x =tf.Variable([36])
with tf.Session() as sess:
    sess.run(x)

    print (x.eval())
