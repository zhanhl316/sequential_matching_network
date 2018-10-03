#!/bin/env/python3
#-*- encoding: utf-8 -*-


import sys
import random
import io
import codecs


reload(sys)
sys.setdefaultencoding('utf-8')

def mix_neg_sample(file_from, file_train, file_valid, file_test):
    '''
    mix negative samples
    :param file_from: file containing only true context-response pairs
    :param file_to: file containing pos:neg = 1:num_neg context-response pairs
    :param num_neg: number of negative samples to coupled with one true sample
    '''
    querys = []
    resps = []
    with open(file_from, 'r', encoding = 'utf-8') as fp_from, \
        open(file_train, 'w', encoding = 'utf-8') as fp_train, \
        open(file_valid, 'w', encoding = 'utf-8') as fp_valid, \
        open(file_test, 'w', encoding = 'utf-8') as fp_test:
        dialogue = []
        for line in fp_from:
            if line == '\n':
                if len(dialogue) > 0:
                    querys.append(dialogue[0:-1])
                    resps.append(dialogue[-1] + '\n')
                dialogue = []
                continue
            dialogue.append(line)

        valid_size = 0
        test_size = 0
        sess_cnt = len(querys)
        for idx in range(sess_cnt):
            if idx < 1000:
                fp_to = fp_valid
                num_neg = 9
            elif idx < 11000:
                fp_to = fp_test
                num_neg = 9
            else:
                fp_to = fp_train
                num_neg = 1
            # dice = random.randint(0, len(resps) - 1)
            # if dice < 1100:
            #     if valid_size < 1000:
            #         fp_to = fp_valid
            #         num_neg = 9
            #         valid_size += 1
            #     else:
            #         fp_to = fp_train
            #         num_neg = 1
            # elif dice < 12000:
            #     if test_size < 10000:
            #         fp_to = fp_test
            #         num_neg = 9
            #         test_size += 1
            #     else:
            #         fp_to = fp_train
            #         num_neg = 1
            # else:
            #     fp_to = fp_train
            #     num_neg = 1

            # write the true context-reponse pair
            fp_to.write(''.join(querys[idx]))
            fp_to.write(resps[idx])

            # write false pairs
            distractor_idx_set = set()
            
            for _ in range(num_neg):
                while True:
                    random_idx = random.randint(0, sess_cnt - 1)
                    if abs(random_idx - idx) > 20 and (random_idx not in distractor_idx_set):
                        fp_to.write(''.join(querys[idx]))
                        fp_to.write(resps[random_idx])
                        distractor_idx_set.add(random_idx)
                        break


def count_session(filepath):
    cnt = 0
    with open(filepath, 'r') as fp:
        for line in fp:
            if line == '\n':
                cnt += 1
    return cnt


def head_session(fin, fout, n):
    '''
    select the first `n` sessions from file `fin` to `fout`
    '''
    cnt = 0
    with open(fin, 'r') as fp_in, open(fout, 'w') as fp_out:
        for line in fp_in:
            fp_out.write(line)
            if line == '\n':
                cnt += 1
                if cnt > int(n)-1:
                    break

def head_sessions(fin, fout, n_begin, n_end):
    '''
    select the n_begin to n_end sessions from file 'fin' to 'fout'
    '''
    cnt = 0

    with codecs.open(fin, 'r', encoding='utf-8') as fp_in, codecs.open(fout, 'w', encoding='utf-8') as fp_out:
        for line in fp_in:
            if line == '\n':
                cnt += 1
                if cnt > int(n_end) - 1:
                    break
            if cnt >= int(n_begin) - 1 and cnt <= int(n_end) - 1:
                cnt +=1
                fp_out.write(line)


def tmp_crop(file_from, file_to):
    '''
    mix negative samples
    :param file_from: file containing only true context-response pairs
    :param file_to: file containing pos:neg = 1:num_neg context-response pairs
    :param num_neg: number of negative samples to coupled with one true sample
    '''
    querys = []
    resps = []
    with open(file_from, 'r') as fp_from, open(file_to, 'w') as fp_to:
        dialogue = []
        for line in fp_from:
            if line == '\n':
                if len(dialogue) > 0:
                    querys.append(dialogue[0:-1])
                    resps.append(dialogue[-1])
                dialogue = []
                continue
            dialogue.append(line)

        num_neg = 9 #num in valid/test file: 9
        for idx in range(10000):
            # write the true context-reponse pair
            fp_to.write(''.join(querys[idx]))
            fp_to.write(resps[idx])
            fp_to.write(str(1) + '\n' + '\n')

            # write false pairs
            distractor_idx_set = []
            for _ in range(num_neg):
                while True:
                    random_idx = random.randint(0, len(resps) - 1)
                    if abs(random_idx - idx) > 20 and (random_idx not in distractor_idx_set):
                        fp_to.write(''.join(querys[idx]))
                        fp_to.write(resps[random_idx])
                        fp_to.write(str(0) + '\n' +'\n')
                        distractor_idx_set.append(random_idx)
                        break
            #fp_to.write('\n')


if __name__ == '__main__':

    #head_session('data/train.txt', 'data/train_3000000.txt', 3000000)
    #head_sessions('data/train.txt', 'data/test_1000.txt', 3000001, 3001001)
    #head_sessions('data/train_pad_comb.csv', 'data/train_test.csv',1,2000000)
    #cnt  = count_session('data/train_test.csv')
    #print str(cnt)
    tmp_crop('data/task5_pad_comb.txt', 'data/task5_mix_pad_comb.txt')

