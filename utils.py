# ----------------------- modules

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation
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

def loadsets(fname, test=False, cols=None):
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

def modellib(name='single'):
    """keras NN model library
    
    Arguments:
        name {str} -- model name
            - 'single': 1-hidden-layer
    
    Return: compiled Keras model
    """
    model = Sequential()

    if name == 'single':
        model.add(Dense(100,input_dim=9216))    # FC1
        model.add(Activation('relu'))
        model.add(Dense(30))                    # FC2
        sgd = SGD(lr=0.01 ,momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error',optimizer=sgd)
    
    return model

def histplot(hist, save=None,  show=True):
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
    fig.subplots_adjust(left=0, right=0.5, bottom=0, top=0.5, hspace=0.05, wspace=0.05)

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
    
