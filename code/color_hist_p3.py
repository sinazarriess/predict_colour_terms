from __future__ import division

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#import wacgen
from collections import Counter,defaultdict
import codecs
from scipy import misc, stats
import scipy
from random import sample
import _pickle as pickle
import gzip
#from train_saia_wac import get_filemax,make_word2feat_dict,make_train,make_subdf,make_wordlist,is_relational,get_refexp
from sklearn import linear_model
import numpy as np
import argparse
import sys
from scipy.spatial import distance as dist
from scipy.stats import entropy
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from keras.utils import np_utils
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from skimage.color import convert_colorspace
from sklearn import linear_model
from skimage.color import rgb2lab
import pandas as pd
import re

basedir = '/Users/sina/generation/ImageData/Corpora/Others/ImageCorpora/SAIA_Data/benchmark/saiapr_tc-12'
colwords = ['blue','red','green','yellow','white','black','grey','pink','purple','orange','brown']

def load_region_data(data_path='../data/',items=[]):
#def load_region_data(feat_path='Xcolorhist3d.npz'):

    
    feat_path = data_path+'Xcolorhist3d_v1.npz'
    X = np.load(feat_path)
    X = X['arr_0']
    print(X.shape)
    
    testindex = np.random.randint(X.shape[0],size=1000)
    trainindex = [x for x in range(X.shape[0]) if not x in testindex]
    # @Kristin: hier müsstest du die Daten so splitten, dass unsere Items im testindex sind!

    Xtrain = X[trainindex]
    Xtest = X[testindex]
    print("Xtrain",Xtrain.shape)
    print("Xtest",Xtest.shape)

    return (Xtrain,Xtest)


def train_region_model(M_train,M_test=[]):
    X_train = M_train[~np.isnan(M_train).any(axis=1)][:,3:]
    y_train = M_train[~np.isnan(M_train).any(axis=1)][:,0]
    Y_train = np_utils.to_categorical(y_train, len(colwords))
    
    if len(M_test) > 0:
        X_test = M_test[~np.isnan(M_test).any(axis=1)][:,3:]
        y_test = M_test[~np.isnan(M_test).any(axis=1)][:,0]
        Y_test = np_utils.to_categorical(y_test, len(colwords))
    
    print('Building model...')
    model = Sequential()
    model.add(Dense(240, input_shape=(X_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(24))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(colwords)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    batch_size = 36
    epochs = 25
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)
    #history = model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0, show_accuracy=True, validation_split=0.1)

    if len(M_test) >0:
        score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
        print('Test score:', score)
        #print('Test accuracy:', score[1])
    
    return model

