from __future__ import division

import matplotlib
import h5py
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
import numpy.lib.recfunctions
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
    with codecs.open(data_path+'image_regions.txt') as f:
        img_reg = f.readlines()
    img_region = [img.split('.')[0] for img in img_reg]
    X = np.load(feat_path)
    X = X['arr_0']
    print(X.shape)

    test_list = []
    for img in img_region:
        for ix in range(len(X)):
            if str(int(X[ix][1]))+'_'+str(int(X[ix][2])) == img:
                test_list.append(X[ix])
    test_list = np.asarray(test_list)
    np.random.seed(0)
    testindex = np.random.randint(X.shape[0],size=970)
    trainindex = [x for x in range(X.shape[0]) if not x in testindex and x not in test_list]

    Xtrain = X[trainindex]
    Xtest = np.concatenate((X[testindex], test_list), axis= 0)
    print("Xtrain",Xtrain.shape)
    print("Xtest",Xtest.shape)

    return (Xtrain,Xtest)

# def reuse_model(M_test=[], saved_model='model.json', saved_weights='model.h5'):
#
#     if len(M_test) > 0:
#         X_test = M_test[~np.isnan(M_test).any(axis=1)][:, 3:]
#         y_test = M_test[~np.isnan(M_test).any(axis=1)][:, 0]
#         Y_test = np_utils.to_categorical(y_test, len(colwords))
#
#     # load json and create model
#     json_file = open(saved_model, 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     loaded_model.load_weights(saved_weights)
#     print("Loaded model from disk")
#
#     score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
#     print('Test score:', score)
#
#     return model

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

    # serialize model to JSON
    #model_json = model.to_json()
    #with open("model.json", "w") as json_file:
    #    json_file.write(model_json)
    # serialize weights to HDF5
    #model.save_weights("model.h5")
    #print("Saved model to disk")

    if len(M_test) >0:
        score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
        print('Test score:', score)
        #print('Test accuracy:', score[1])

    return model

def predict_color(model, M_test):
    X_test = M_test[~np.isnan(M_test).any(axis=1)][:, 3:]
    y_test = M_test[~np.isnan(M_test).any(axis=1)][:, 0]
    Y_test = np_utils.to_categorical(y_test, len(colwords))
    M_anno = []
    batch_size = 36

    pred = model.predict(X_test, batch_size = batch_size, verbose= 0)
    X_test_anno = np.hstack([pred, X_test])
    M_anno.append(X_test_anno)

    M_anno = np.vstack(M_anno)
    print("Collection Test Splits", M_anno.shape)

    return M_anno