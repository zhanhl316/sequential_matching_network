#!/bin/env python
#-*- encoding:utf-8 -*-


import numpy as np
import pandas as pd
import codecs
import u8_csv_utils


def convert(in_fpath, out_fpath):
    dump_list = []
    with codecs.open(in_fpath, 'r') as rfd:
        reader = u8_csv_utils.UnicodeReader(rfd)
        # skip header line
        next(reader)
        for line in reader:
            question = line[1].strip('\n')
            cluster_id = line[2]
            dump = '%s\t%s' % (cluster_id, question)
            dump_list.append(dump)
            # dump  
    with codecs.open(out_fpath, 'w', 'utf-8') as wfd:
        wfd.write('\n'.join(dump_list)) 

if '__main__'==__name__:
    in_fpath = './abc.csv'
    out_fpath = './abc.txt'
    convert(in_fpath, out_fpath)

