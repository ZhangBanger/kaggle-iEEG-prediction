import numpy as np
import tensorflow as tf

from preprocess import WINDOW_SIZE, CHANNELS
from util import weight_variable, bias_variable

BATCH_SIZE = 256

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, WINDOW_SIZE, CHANNELS])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("layer1"):
    filter_weights = weight_variable([1, CHANNELS, CHANNELS])
    feature_map = tf.nn.conv1d(x, filter_weights, stride=1, padding='SAME')
    activation = tf.nn.elu(feature_map + bias_variable([WINDOW_SIZE, CHANNELS]))
    dropout = tf.nn.dropout(activation, keep_prob=keep_prob)

with tf.variable_scope("layer4"):
    flattened = tf.reshape(activation, [-1, WINDOW_SIZE * CHANNELS])
    weights = weight_variable([WINDOW_SIZE * CHANNELS, 1])
    bias = bias_variable([1])
    y_conv = tf.matmul(flattened, weights) + bias

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

optimizer = tf.train.AdamOptimizer(3e-4)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_step = optimizer.minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

for iteration in range(20000):
    xs = []
    ys = []
    # Replace this with file reader code
    # https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html#file-formats
    # for j in range(BATCH_SIZE):
        # data, label, meta = next(sample_generator)
        # xs.append(data)
        # ys.append(label)

    xs = np.dstack(xs).transpose((2, 0, 1))
    ys = np.vstack(ys)

    _, train_loss, train_accuracy = sess.run(
        [train_step, cross_entropy, accuracy],
        feed_dict={x: xs, y_: ys, keep_prob: 0.75}
    )

    print("step %d, training accuracy %g, train loss %.4f" % (iteration, train_accuracy, train_loss))
