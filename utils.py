#!/usr/bin/env python

"""Utils by amblizer, for kaggle facial keypoints
   detection with Keras, updated: 2018-04-21
"""

# ----------------------- modules

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import numpy as np
import os
import random

# ----------------------- helper function

def normlabel(y, reverse=False):
    """normalize / de-normalize label 
    
    Arguments:
        y {np.array} -- raw or normalized label
    
    Keyword Arguments:
        reverse {bool} -- normalize / de-normalize label (default: {False})
    """
    if reverse:
        return  y*48 + 48
    else:
        return (y - 48) / 48

def loadset(fname, test=False, cols=None):
    """Load datasets from CSV file
    
    Arguments:
        fname {str} -- filepath
    
    Keyword Arguments:
        test {bool} -- if file is test set (default: {False})
        cols {list} -- retain 'Image' and designated columns only (default: {None})
    
    Return:(X, y)
    """
    df = read_csv(fname)    # pixel value stored in 'Image' column
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:
        df = df[list(cols) + ['Image']]

    print(df.count())       # summary
    df = df.dropna()        # holds rows with all keypoints only

    X = np.vstack(df['Image'].values) / 255.    # normalize grayscale in[0,1]
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values          # columns without 'image'
        y = normlabel(y, reverse=False)         # normalize coords in [0,1]
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    return (X, y)

def loadset2d(fname, test=False, cols=None):
    (X, y) = loadset(fname, test=False, cols=None)
    X = X.reshape(-1, 96, 96, 1)
    return (X, y)

def modellib(name='single', dim=9216):
    """keras NN model library
    
    Arguments:
        name {str} -- model name
            - 'single': 1-hidden-layer
        dim {int/tuple} -- input dimension (default: {9216})

    Return: compiled Keras model
    """
    model = Sequential()

    if name == 'single':
        model.add(Dense(100,input_dim=dim))    # FC1
        model.add(Activation('relu'))
        model.add(Dense(30))                    # FC2
        sgd = SGD(lr=0.01 ,momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error',optimizer=sgd)
    
    if name == 'CNN':
        model.add(Conv2D(32, (3, 3), input_shape=dim))  # CONV1
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))  # FC1

        model.add(Conv2D(64, (2, 2)))           # Conv2
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))  # FC2

        model.add(Conv2D(128, (2, 2)))          # Conv3
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))  # FC3

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(30))                    # 30 coords

        sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=sgd)

    return model

def histplot(hist, save=None, show=True):
    """plotting loss function
    
    Arguments:
        hist {keras.callbacks.History} -- hist = model.fit
    
    Keyword Arguments:
        save {str} -- filename if to save (default: {None})
        show {bool} -- show plot (default: {True})
    """
    plt.figure()
    plt.plot(hist.history['loss'], linewidth=2, label='train')
    plt.plot(hist.history['val_loss'], linewidth=2,label='valid set')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.ylim(1e-3, 1e-2)
    plt.yscale('log')
    plt.grid()               # show metrics(0, 20, 40...)
    plt.legend(loc='best')
    figure = plt.gcf()

    if save != None:
        figure.savefig(save, dpi=300)
        print(save, 'saved.')

    if show:
        plt.show()

def histplotdiff(hist1, hist2, save=None, show=True):
    """plotting loss function
    
    Arguments:
        hist {keras.callbacks.History} -- hist = model.fit
    
    Keyword Arguments:
        save {str} -- filename if to save (default: {None})
        show {bool} -- show plot (default: {True})
    """
    plt.figure()
    plt.plot(hist1.history['loss'], linewidth=2, label='train1')
    plt.plot(hist1.history['val_loss'], linewidth=2,label='valid1')
    plt.plot(hist2.history['loss'], linewidth=2, label='train2')
    plt.plot(hist2.history['val_loss'], linewidth=2,label='valid2')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.ylim(1e-3, 1e-2)
    plt.yscale('log')
    plt.grid()               # show metrics(0, 20, 40...)
    plt.legend(loc='best')
    figure = plt.gcf()

    if save != None:
        figure.savefig(save, dpi=300)
        print(save, 'saved.')

    if show:
        plt.show()

def predplot(xtest, ypred, save=None, show=True):
    """plotting pred outcome
    
    Arguments:
        xtest {np.array} -- test img
        ypred {np.array} -- pred label
    
    Keyword Arguments:
        save {str} -- filename if to save (default: {None})
        show {bool} -- show plot (default: {True})
    """
    total = xtest.shape[0]
    start = random.randint(0, total-16)
    print('image',start,'to',start+16)

    fig = plt.figure(figsize=(12, 12))
    # fig.subplots_adjust(left=0, right=0.5, bottom=0, top=0.5, hspace=0.05, wspace=0.05)

    for i in range(start, start+16):
        x, y = xtest[i], ypred[i]
        img  = x.reshape(96, 96)
        axis = fig.add_subplot(4, 4, i-start+1, xticks=[], yticks=[])
        axis.imshow(img, cmap='gray')   # show image
        axis.scatter(normlabel(y[0::2], reverse=True),  # show lanmark
                     normlabel(y[1::2], reverse=True), marker='x', s=10)
    figure = plt.gcf()
    
    if save != None:    # deprecated
        figure.savefig(save, dpi=300)
        print(save, 'saved.')

    if show:
        plt.show()

def predplotdiff(xtest, ypred1, ypred2, save=None, show=True):
    """compare pred outcome
    
    Arguments:
        xtest {np.array} -- test img
        ypred {np.array} -- pred label
    
    Keyword Arguments:
        save {str} -- filename if to save (default: {None})
        show {bool} -- show plot (default: {True})
    """
    total = xtest.shape[0]
    start = random.randint(0, total-16)
    print('image',start,'to',start+4)

    # fig.subplots_adjust(left=0, right=0.5, bottom=0, top=0.5, hspace=0.05, wspace=0.05)

    fig = plt.figure(figsize=(12, 6))
    for n,i in enumerate(range(start, start+4)):
        x, y1, y2 = xtest[i], ypred1[i], ypred2[i]
        img  = x.reshape(96, 96)
        
        axis1 = fig.add_subplot(2, 4, 2*n+1, xticks=[], yticks=[])
        axis1.imshow(img, cmap='gray')   # show image
        axis1.scatter(normlabel(y1[0::2], reverse=True),  # show lanmark
                      normlabel(y1[1::2], reverse=True), color='b', marker='x', s=10, label='pred1')
        axis1.legend(loc='best')
        
        axis2 = fig.add_subplot(2, 4, 2*n+2, xticks=[], yticks=[])
        axis2.imshow(img, cmap='gray')   # show image
        axis2.scatter(normlabel(y2[0::2], reverse=True),  # show lanmark
                      normlabel(y2[1::2], reverse=True), color='r', marker='x', s=10, label='pred2')
        axis2.legend(loc='best')
                
    figure = plt.gcf()
    
    if save != None:    # deprecated
        figure.savefig(save, dpi=300)
        print(save, 'saved.')

    if show:
        plt.show()

def savemodel(model, name='single_hidden_layer', toprint=True):
    """save model to file
    
    Arguments:
        model {keras.models.Sequential} -- keras model
    
    Keyword Arguments:
        name {str} -- file name to save (default: {'single_hidden_layer'})
    """
    json_string = model.to_json()
    open(name+'_architecture.json', 'w').write(json_string)  # structure
    model.save_weights(name+'_weights.h5')                   # weights
    
    if toprint:
        print('Structure:', name+'_architecture.json')
        print('Weights:', name+'_weights.h5')

def loadmodel(frame, weights, toprint=True):
    """load model from file
    
    Arguments:
        frame {str} -- NN structure
        weights {str} -- NN weights
    
    Keyword Arguments:
        toprint {bool} -- detailed (default: {True})

    Return: keras.models.Sequential
    """
    model = model_from_json(open(frame).read())
    model.load_weights(weights)
    
    if toprint:
        print('Structure from:', frame)
        print('Weights from:', weights)
    
    return model