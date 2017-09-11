# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
import keras
import re
import datetime
from pandas import Series, DataFrame
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential, load_model
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
            istr += ' '
            istr += row[7]
        if not istr:
            print('error')
            print(row[0])
            exit()
        textlist.append(istr)       
        labellist.append(int(row[2]))
    return textlist, labellist

def get_test_data(filename):
    df = pd.read_csv(filename)
    textlist = []
    idlist = []
    for index, row in df.iterrows():
        istr = ''
        if isinstance(row[3], str):
            istr += row[3]
        if isinstance(row[6], str):
            istr += ' '
            istr += row[6]
        if not istr:
            print('error')
            print(row[0])
            exit()
        textlist.append(istr)       
        idlist.append(int(row[1]))
    return textlist, idlist

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

def get_hd5_model():
    model = load_model('mymodel.h5')
    return model

if __name__=='__main__':
    filename = 'case_full.csv'
    textlist, labellist = get_data(filename)
    testfilename = 'question_full.csv'
    t_textlist,idlist = get_test_data(testfilename)
    pprint(t_textlist[:2])
    pprint(idlist[:2])
    # textlist = textlist[:20]
    # labellist = labellist[:20]
    text_array = get_text_array(textlist)
    t_text_array = get_text_array(t_textlist)
    full_text_array = []
    full_text_array.extend(text_array)
    full_text_array.extend(t_text_array)
    text_num_map = get_text_num_map(full_text_array)
    numpy_array = get_num_array(text_array, text_num_map)
    t_numpy_array = get_num_array(t_text_array, text_num_map)
    numpy_array = list(sequence.pad_sequences(numpy_array, maxlen=50))
    t_numpy_array = list(sequence.pad_sequences(t_numpy_array, maxlen=50))
    train_x = np.array(numpy_array)
    t_x = np.array(t_numpy_array)
    train_y = keras.utils.to_categorical(labellist, num_classes=13)
    # pprint(text_array[:2])
    # pprint(train_y[:2])
    x1 = train_x[::2]
    y1 = train_y[::2]
    x2 = train_x[1::2]
    y2 = train_y[1::2]
    llen = len(text_num_map)+1
    model = get_lstm_model(llen)
    model.fit(train_x, train_y, batch_size=16, epochs=20)
    # model = get_hd5_model()
    # score = model.evaluate(x2, y2, batch_size=16)
    classes = model.predict_classes(t_x)
    # print(score)
    # print(classes)
    da = DataFrame(idlist)
    da[1] = classes
    da.to_csv('result.csv',sep=',', encoding='utf-8',index=False,header=False)
    model.save('mymodel.h5')
    
    
    

    



