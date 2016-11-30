import numpy as np
import tensorflow as tf


def gauss_kern_1d(size, factor=1.5):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    x = np.arange(-int(size * factor), int(size * factor) + 1)
    g = np.exp(-(x ** 2 / size))
    return g / g.sum()


def subsample(x, channels, rate):
    rate = int(rate)
    x_flatten = x.reshape(-1)
    x_convolve = np.convolve(x_flatten, gauss_kern_1d(2 / 3 * rate), mode="same")
    x_subsample = x_convolve[::rate]
    return x_subsample.reshape(-1, channels)


def normalize(x):
    # Hacky normalization based on eyeball
    return x / 32.


def weight_variable(shape, name="W"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name="b"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)
