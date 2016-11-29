# from keras.layers import Embedding, Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D

from keras.layers import (Dropout, Activation, Dense, Convolution1D, Convolution2D, MaxPooling2D, Flatten)
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

# create the model
batch_size = 256
channels = 16
sequence_len = 1000
embed_dim = 128
# input_dim = 3
# sequence_len = 50
# embed_dim = 8
drop_prob = 0.25

input_shape = (sequence_len, channels)

print('Build model...')
mo = Sequential()

nfilt1 = 16

mo.add(Convolution1D(nb_filter=nfilt1, filter_length=1,
                     input_dim=channels, input_length=sequence_len,
                     W_regularizer='l1l2',
                     ))
# output: 3D tensor with shape: (samples, new_steps, nb_filter). steps value might have changed due to padding.
mo.add(Reshape((1, nfilt1, sequence_len)))
mo.add(BatchNormalization())
mo.add(Activation('elu'))
mo.add(Dropout(drop_prob))

mo.add(Convolution2D(nb_filter=4, nb_row=2, nb_col=8,
                     W_regularizer='l1l2', ))
mo.add(BatchNormalization())
mo.add(ZeroPadding2D(padding=(1, 4)))
mo.add(Activation('elu'))
mo.add(MaxPooling2D((2, 4)))
mo.add(Dropout(drop_prob))

mo.add(Convolution2D(nb_filter=4, nb_row=8, nb_col=4,
                     W_regularizer='l1l2', ))
mo.add(BatchNormalization())
mo.add(ZeroPadding2D(padding=(1, 4)))
mo.add(Activation('elu'))
mo.add(MaxPooling2D((2, 4)))
mo.add(Dropout(drop_prob))

#################
print(mo.summary())

mo.add(Flatten())
mo.add(Dense(1))
mo.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
mo.compile(loss='binary_crossentropy',
           optimizer='adam',
           metrics=['accuracy'])

# print(model.summary())
# # model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64)
