import numpy as np
import tensorflow as tf

from preprocess import WINDOW_SIZE, CHANNELS
from util import weight_variable, bias_variable

BATCH_SIZE = 128
WINDOW_SIZE = 100
CHANNELS = 16

CHANNELS_L1 =  32
CHANNELS_L2 = 4
CHANNELS_L3 = 2

# dim order:   x, y, c_in, c_out
maxpool_ksize = [1, 2, 4, 1]
KERNEL2 = [30, 2, 1, CHANNELS_L2]
KERNEL3 = [8, 4, CHANNELS_L2, CHANNELS_L3]

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, WINDOW_SIZE, CHANNELS])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("layer1"):
    filter_weights = weight_variable([1, CHANNELS, CHANNELS_L1], name="weights")
    feature_map = tf.nn.conv1d(x, filter_weights, stride=1, padding='SAME')
    print("layer1/feature_map", feature_map.get_shape())
    activation = tf.nn.elu(feature_map + bias_variable(feature_map.get_shape()[1:]))
    activation = tf.nn.dropout(activation, keep_prob=keep_prob)
    activation = tf.reshape(activation, [BATCH_SIZE,  CHANNELS_L1,  WINDOW_SIZE, 1])
    print("layer1", activation.get_shape())

with tf.variable_scope("layer2"):
    print("KERNEL2", KERNEL2)
    filter_weights = weight_variable(KERNEL2, name="weights")
    feature_map = tf.nn.conv2d(activation, filter_weights, strides=[1,1,1,1], padding='SAME')
    activation = tf.nn.elu(feature_map + bias_variable(feature_map.get_shape()[1:]))
    activation = tf.nn.max_pool(activation, maxpool_ksize, [1,1,1,1], padding='VALID',
                                data_format='NHWC', name='maxpool')
    activation = tf.nn.dropout(activation, keep_prob=keep_prob)
    print("layer2", activation.get_shape())

with tf.variable_scope("layer3"):
    print("KERNEL3", KERNEL3)
    filter_weights = weight_variable(KERNEL3, name="weights")
    feature_map = tf.nn.conv2d(activation, filter_weights, strides=[1,1,1,1], padding='SAME')
    activation = tf.nn.elu(feature_map + bias_variable(feature_map.get_shape()[1:]))
    activation = tf.nn.max_pool(activation, maxpool_ksize, [1,1,1,1], padding='VALID',
                                data_format='NHWC', name='maxpool')
    activation = tf.nn.dropout(activation, keep_prob=keep_prob)
    print("layer3", activation.get_shape())

with tf.variable_scope("layer4"):
    dim = np.prod(activation.get_shape().as_list()[1:])
    print(dim)
    flattened = tf.reshape(activation, [-1, dim])
    weights = weight_variable([dim, 1])
    bias = bias_variable([1])
    y_conv = tf.matmul(flattened, weights) + bias

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

optimizer = tf.train.AdamOptimizer(3e-4)
gradients = optimizer.compute_gradients(cross_entropy)
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
    # reader = tf.RecordReader()
    #xs = np.dstack(xs).transpose((2, 0, 1))
    #ys = np.vstack(ys)
    xs = np.random.randn(*[BATCH_SIZE, WINDOW_SIZE, CHANNELS])
    ys = np.random.binomial(1,0.2, BATCH_SIZE).reshape(-1, 1)

    _, train_loss, train_accuracy = sess.run(
        [train_step, cross_entropy, accuracy],
        feed_dict={x: xs, y_: ys, keep_prob: 0.75}
    )

    print("step %d, training accuracy %g, train loss %.4f" % (iteration, train_accuracy, train_loss))
