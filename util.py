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
    # 32 is the magic number!
    return x / 32.


def weight_variable(shape, name="W"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name=name, initializer=initial)


def bias_variable(shape, name="b"):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name=name, initializer=initial)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + var.name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + var.name, stddev)
        tf.scalar_summary('max/' + var.name, tf.reduce_max(var))
        tf.scalar_summary('min/' + var.name, tf.reduce_min(var))
        tf.scalar_summary('l2norm/' + var.name, tf.nn.l2_loss(var))
        tf.scalar_summary('l1norm/' + var.name, tf.reduce_sum(tf.abs(var)))
        tf.histogram_summary(var.name, var)
