#!/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import sys 

reload(sys)
sys.setdefaultencoding('utf-8')

import csv
import os 
import io

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, base_dir)



def combine_to_lines(file_from, file_to):
    
    with io.open(file_from, 'r', encoding='utf-8') as fp_from, io.open(file_to, 'w', encoding='utf-8') as fp_to:
        querys = []
        respos = []
        dialogue = []
        for line in fp_from.readlines():
            if line != '\n':
                dialogue.append(line.strip('\n') + ' ' )
            if line == '\n':
                querys.append(dialogue[:-1])
                respos.append(dialogue[-1] + '\n')
                fp_to.write(unicode(''.join(querys[0])))
                fp_to.write(u'\n')
                fp_to.write(respos[0] + '\n')
                querys = []
                respos = []
                dialogue = []
                continue


def combine_to_row(file_from, file_to):
    
    with open(file_from, 'rb') as fp_from, open(file_to, 'w') as fp_to:
        querys = []
        respos = []
        labels = []
        dialogue = [] 
        writeCSV = csv.writer(fp_to, dialect=("excel"))
        
        for line in fp_from.readlines():
            if line != '\n':
                dialogue.append(line.strip('\n'))
            if line =='\n':
                writeCSV.writerow([dialogue[0], dialogue[1], dialogue[2], dialogue[3],dialogue[4],dialogue[5],dialogue[6],dialogue[7],dialogue[8],dialogue[9],dialogue[10]])
                #writeCSV.writerow([dialogue[0], dialogue[1], dialogue[-1]]) 
                dialogue = []
                continue


def txt2csv(inputfile,outputfile):  
    datacsv = open(outputfile,'w')  
    csvwriter = csv.writer(datacsv,dialect=("excel"))  
    mainfileH = open(inputfile,'rb')  
    for line in mainfileH.readlines():     
        #print "Debug: " + line.replace('\n','')      
        csvwriter.writerow([line])  
    datacsv.close()  
    mainfileH.close()

def forTask5(inputfile, outputfile):
    with open(inputfile, 'r') as in_f, open(outputfile, 'w') as out_f:
        cnt = 0
        dialogue = []
        index = 0
        for line in in_f.readlines():
            index += 1 
            if (index == 1) or (index % 3 == 2):
                dialogue.append(line)
            if line == '\n':
                cnt += 1
                if cnt == 10:
                    cnt = 0
                    out_f.write(''.join(dialogue))
                    out_f.write('\n')
                    dialogue = []
                    index = 0


if __name__ == '__main__':
    
    #forTask5('./data/task5_pad_comb.txt', './data/task5_mix_pad_comb.txt')
    #combine_to_lines('./data/task5_pad.txt', './data/task5_pad_comb.txt')

    combine_to_row('./data/task5_mix_pad_comb.txt', './data/task5_pad_comb.csv')

