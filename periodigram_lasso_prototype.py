
# coding: utf-8
import pandas as pd
import six
import os
import numpy as np
from pandas import HDFStore
from keras import backend as K
from keras.objectives import mean_squared_error
def log_poisson(y_true, log_y_pred):
    return K.mean(K.exp(log_y_pred) - y_true * log_y_pred, axis=-1)

def poi_gau_mix(y_true, log_y_pred):
    return log_poisson(y_true, log_y_pred) + 0.01*mean_squared_error(y_true, K.exp(log_y_pred))


from tensorflow import python_io, train
import tensorflow as tf


# from keras.layers import Embedding, Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Dense, Dropout, Activation, Embedding, Input
from keras.models import Model
from keras.constraints import MaxNorm
from keras.layers.advanced_activations import ELU

from keras.models import Model, Sequential
from keras.layers import Dense, LSTM
from keras.regularizers import l2, l1l2

# create the model
inp_channels = 16
input_dim = 120001*inp_channels
# embed_dim = 128


############################################################################

from keras.callbacks import LearningRateScheduler
def get_label(infile):
    return infile.split("/")[-1].split(".")[0][-1] == "0"


def read_periodigrams(h5filename = "data/periodigrams.h5", BATCH_SIZE=1):
    xlist = []
    ylist = []
    h5 = HDFStore(h5filename)
    for nn, kk in enumerate(h5.keys()):
        yy = get_label(kk)
        xx = h5[kk].as_matrix()
        xlist.append(xx.T.ravel())
        ylist.append(yy)
        if ((nn+1) % BATCH_SIZE == 0):
            yield np.vstack(xlist), np.vstack(ylist)
            xlist = []
            ylist = []


############################################################################
def train_model_l1(l1penalty):
    print('Build model...')
    mo = Sequential()
    mo.add(Dense(1, input_shape=(input_dim,),  W_regularizer=l1l2(l1=l1penalty, l2=l1penalty)))
    mo.add(Activation('sigmoid'))

    # try using different optimizers and different optimizer configs
    mo.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    def scheduler(epoch):
        if epoch == 15:
            mo.optimizer.lr.set_value(.01)
        if epoch == 20:
            mo.optimizer.lr.set_value(.005)
        return float(mo.optimizer.lr.get_value())

    change_lr = LearningRateScheduler(scheduler)

    ############################################################################
    filepath = "periodigram_lasso"
    # mo.load_weights(filepath)
    mo.optimizer.lr.set_value(.01)


    mo.fit_generator(gen, nb_worker=1,
                     nb_epoch=10, samples_per_epoch = samples_per_epoch,
                    callbacks=[change_lr])
    ####################################################
    print("MODEL DONE")
    return mo


#for xx, yy in read_periodigrams(h5filename = "data/periodigrams.h5"):
#    print(xx.shape, yy)

from itertools import cycle

datadir = "data/"
h5filename = "data/periodigrams.h5"
BATCH_SIZE = 4
samples_per_epoch = 8
"remove cycle in the real set"
gen = cycle(read_periodigrams(h5filename, BATCH_SIZE = 2))
####################################################

step = 0.5
l1penaltylist = 10.0**np.arange(-4,-1, step)

for l1penalty in l1penaltylist:
    print("l1penalty=", l1penalty)
    mo = train_model_l1(l1penalty)
    la = mo.layers[0]
    W = np.reshape( la.get_weights()[0], (-1,16))
    pd.DataFrame(W, columns=np.arange(inp_channels)).\
                to_csv("weights_neglog_penalty_%.2f.tab" % -np.log10(l1penalty), sep="\t")
    #mo.
"reshape weights in following way:"
# np.reshape(xx[0],(-1,16))[ freq, channel]

#la.get_weights()[0].reshape(inp_channels,-1)[0].shape

