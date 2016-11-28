from itertools import cycle
import os

import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.objectives import mean_squared_error
from keras.optimizers import Adam
from keras.regularizers import l1l2
from pandas import HDFStore

# create the model
inp_channels = 16
input_dim = 120001 * inp_channels


def log_poisson(y_true, log_y_pred):
    return K.mean(K.exp(log_y_pred) - y_true * log_y_pred, axis=-1)


def poi_gau_mix(y_true, log_y_pred):
    return log_poisson(y_true, log_y_pred) + 0.01 * mean_squared_error(y_true, K.exp(log_y_pred))


############################################################################


def get_label(infile):
    return infile.split("/")[-1].split(".")[0][-1] == "0"


def read_periodograms(file_path, batch_size=1):
    x_batch = []
    y_batch = []
    h5 = HDFStore(file_path)
    for nn, kk in enumerate(h5.keys()):
        yy = get_label(kk)
        xx = h5[kk].as_matrix()
        x_batch.append(xx.T.ravel())
        y_batch.append(yy)
        if (nn + 1) % batch_size == 0:
            yield np.vstack(x_batch), np.vstack(y_batch)
            x_batch = []
            y_batch = []


############################################################################
def train_model_l1(l1penalty):
    print('Build model...')
    model = Sequential()
    model.add(Dense(1, input_shape=(input_dim,), W_regularizer=l1l2(l1=l1penalty, l2=l1penalty)))
    model.add(Activation('sigmoid'))
    adam_optimizer = Adam(lr=0.01, decay=0.99)

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer=adam_optimizer,
                  metrics=['accuracy'])

    ############################################################################
    model.fit_generator(gen, nb_worker=1,
                        nb_epoch=10, samples_per_epoch=samples_per_epoch)
    ####################################################
    print("MODEL DONE")
    return model

data_dir = os.path.expanduser("~/data/seizure-prediction")
hdf5_path = os.path.join(data_dir, "periodograms.h5")
BATCH_SIZE = 128
samples_per_epoch = 128 * 48
"remove cycle in the real set"
gen = cycle(read_periodograms(hdf5_path, batch_size=BATCH_SIZE))
####################################################

step = 0.5
l1_penalties = np.array(10.0 ** np.arange(-4, -1, step))

for l1_penalty in l1_penalties:
    print("l1penalty=", l1_penalty)
    penalized_model = train_model_l1(l1_penalty)
    la = penalized_model.layers[0]
    W = np.reshape(la.get_weights()[0], (-1, 16))
    pd.DataFrame(W, columns=np.arange(inp_channels)). \
        to_csv("weights_neglog_penalty_%.2f.tab" % -np.log10(l1_penalty), sep="\t")
"reshape weights in following way:"
# np.reshape(xx[0],(-1,16))[ freq, channel]

# la.get_weights()[0].reshape(inp_channels,-1)[0].shape
