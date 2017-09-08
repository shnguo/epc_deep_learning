# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
import keras
import re
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from pprint import pprint

Filters=' !"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
pattern = re.compile(r'[a-z0-9]',re.I)

def get_data(filename):
    df = pd.read_csv(filename)
    textlist = []
    labellist = []
    for index, row in df.iterrows():
        istr = ''
        if isinstance(row[4], str):
            istr += row[4]
        if isinstance(row[7], str):
            istr += row[7]
        if not istr:
            print('error')
            print(row[0])
            exit()
        textlist.append(istr)       
        labellist.append(int(row[2]))
    return textlist, labellist

def get_text_array(textlist):
    text_array = []
    for _t in textlist:
        _tlist = list(jieba.cut(_t))
        _tlist = list(filter(lambda x: x and x not in Filters and not pattern.findall(x), _tlist))
        text_array.append(_tlist)
    return text_array

def get_text_num_map(text_array):
    _textlist = []
    text_num_map = {}
    for _list in text_array:
        _textlist.extend(_list)
    _textlist = list(set(_textlist))
    _textlist.sort()
    length = len(_textlist)
    for i in range(0,length):
        text_num_map[_textlist[i]] = i+1
    return text_num_map

def get_num_array(text_array, text_num_map):
    numpy_array = []
    for _list in text_array:
        numpy_array.append(list(map(lambda x:text_num_map[x], _list)))
    return numpy_array

def get_lstm_model(length):
    model = Sequential()
    model.add(Embedding(length, 256))
    model.add(LSTM(256,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(13, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    return model

def main():
    model.fit(xx, yy, batch_size=16, nb_epoch=10)
    score = model.evaluate(xxx, yyy, batch_size=16)
    print(score)


if __name__=='__main__':
    filename = 'case_full.csv'
    textlist, labellist = get_data(filename)
    text_array = get_text_array(textlist)
    text_num_map = get_text_num_map(text_array)
    numpy_array = get_num_array(text_array, text_num_map)
    numpy_array = list(sequence.pad_sequences(numpy_array, maxlen=50))
    train_x = np.array(numpy_array)
    train_y = keras.utils.to_categorical(labellist, num_classes=13)
    pprint(train_x[:2])
    pprint(train_y[:2])
    x1 = train_x[::2]
    y1 = train_y[::2]
    x2 = train_x[1::2]
    y2 = train_y[1::2]
    llen = len(text_num_map)+1
    model = get_lstm_model(llen)
    model.fit(x1, y1, batch_size=16, nb_epoch=10)
    score = model.evaluate(x2, y2, batch_size=16)
    print(score)

    



