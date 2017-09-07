# -*- coding: utf-8 -*-
import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
import keras
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from pprint import pprint

Filters=' !"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

if __name__=='__main__':
    text = []
    label = []
    labels = []
    with open('case-data.txt') as f:
        for line in f:
            templine = line.strip('\n').split('||')
            text.append(templine[3]+' '+templine[6])
            label.append([int(templine[1])])
            labels.append(int(templine[1]))
    ll = len(set(labels))
    print(ll)
    print(len(text))
    print(len(label))
    # text = text[:100]
    # label = label[:100]
    af_label = keras.utils.to_categorical(label, num_classes=13)
    text_list = []
    for _t in text:
        tmp_list = list(jieba.cut(_t,HMM=False))
        tmp_list = list(filter(lambda x: x!='' and x not in Filters and x is not None, tmp_list))
        text_list.append(tmp_list)
    # d2v_train = pd.DataFrame(text_list)
    # # print(d2v_train.head())
    w = []
    for  _series in text_list:
        w.extend(_series)
    w = list(set(w))
    w = sorted(w)
    wlen = len(w)
    dic = {}
    # pprint(w)
    for i in range(wlen):
        dic[w[i]]=i+1

    # dic = pd.DataFrame(pd.Series(w).value_counts())
    # dic.sort_index()
    # dic['id']=list(range(1,len(dic)+1))
    # pprint(dic.head())
    text_num_list = []
    for _textlist in text_list:
        text_num_list.append(list(map(lambda x:dic[x], _textlist)))
    af_text_num_list = list(sequence.pad_sequences(text_num_list, maxlen=50))
    x = np.array(af_text_num_list)
    y = af_label
    xx = x[:-10]
    yy = y[:-10]
    xxx = x[-10:]
    yyy = y[-10:]
    pprint(x[:5])
    pprint(y[:5])
    print('Build model...')
    model = Sequential()
    model.add(Embedding(len(dic)+1, 256))
    model.add(LSTM(256,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(13, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    model.fit(xx, yy, batch_size=16, nb_epoch=10)
    score = model.evaluate(xxx, yyy, batch_size=16)
    print(score)



