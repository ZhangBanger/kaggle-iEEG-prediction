import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.python.ops import control_flow_ops

from preprocess import from_example_proto
from util import weight_variable, bias_variable

NUM_EPOCHS = 10
BATCH_SIZE = 128
WINDOW_SIZE = 1000
CHANNELS = 16

CHANNELS_L1 = 32
CHANNELS_L2 = 4
CHANNELS_L3 = 2

# dim order:   x, y, c_in, c_out
maxpool_ksize = [1, 2, 4, 1]
KERNEL2 = [32, 2, 1, CHANNELS_L2]
KERNEL3 = [8, 4, CHANNELS_L2, CHANNELS_L3]
scale_bn = False
decay_bn = 0.999
epsilon_bn = 0.001


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    example, label = from_example_proto(serialized_example)

    return example, label


def input_pipeline(read_threads=32, train=True):
    data_dir = os.path.expanduser("~/data/seizure-prediction/preprocessed")

    file_suffix = ".train" if train else ".valid"
    filename_list = list(filter(lambda x: x.endswith(file_suffix), os.listdir(data_dir)))
    filename_queue = tf.train.string_input_producer(filename_list, num_epochs=NUM_EPOCHS)
    example_list = [read_and_decode(filename_queue) for _ in range(read_threads)]

    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 1024
    capacity = min_after_dequeue + read_threads * BATCH_SIZE
    example_batch, label_batch = tf.train.shuffle_batch_join(
        example_list,
        batch_size=BATCH_SIZE,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
    )
    return example_batch, label_batch


with tf.name_scope("input"):
    x, y_ = input_pipeline()

keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("layer1"):
    filter_weights = weight_variable([1, CHANNELS, CHANNELS_L1], name="weights")
    feature_map = tf.nn.conv1d(x, filter_weights, stride=1, padding='SAME')
    print("layer1/feature_map", feature_map.get_shape())
    feature_map = batch_norm(feature_map, decay=decay_bn, center=True, scale=scale_bn,
                             epsilon=epsilon_bn, activation_fn=None)
    activation = tf.nn.elu(feature_map + bias_variable(feature_map.get_shape()[1:]))
    activation = tf.nn.dropout(activation, keep_prob=keep_prob)
    activation = tf.reshape(activation, [BATCH_SIZE, CHANNELS_L1, WINDOW_SIZE, 1])
    print("layer1", activation.get_shape())

with tf.variable_scope("layer2"):
    print("KERNEL2", KERNEL2)
    filter_weights = weight_variable(KERNEL2, name="weights")
    feature_map = tf.nn.conv2d(activation, filter_weights, strides=[1, 1, 1, 1], padding='SAME')
    feature_map = batch_norm(feature_map, decay=decay_bn, center=True, scale=scale_bn,
                             epsilon=epsilon_bn, activation_fn=None)

    activation = tf.nn.elu(feature_map + bias_variable(feature_map.get_shape()[1:]))
    activation = tf.nn.max_pool(activation, maxpool_ksize, [1, 1, 1, 1], padding='VALID',
                                data_format='NHWC', name='maxpool')
    activation = tf.nn.dropout(activation, keep_prob=keep_prob)
    print("layer2", activation.get_shape())

with tf.variable_scope("layer3"):
    print("KERNEL3", KERNEL3)
    filter_weights = weight_variable(KERNEL3, name="weights")
    feature_map = tf.nn.conv2d(activation, filter_weights, strides=[1, 1, 1, 1], padding='SAME')
    feature_map = batch_norm(feature_map, decay=decay_bn, center=True, scale=scale_bn,
                             epsilon=epsilon_bn, activation_fn=None)

    activation = tf.nn.elu(feature_map + bias_variable(feature_map.get_shape()[1:]))
    activation = tf.nn.max_pool(activation, maxpool_ksize, [1, 1, 1, 1], padding='VALID',
                                data_format='NHWC', name='maxpool')
    activation = tf.nn.dropout(activation, keep_prob=keep_prob)
    print("layer3", activation.get_shape())

with tf.variable_scope("layer4"):
    dim = np.prod(activation.get_shape().as_list()[1:])
    print(dim)
    flattened = tf.reshape(activation, [-1, dim])
    weights = weight_variable([dim, 1])
    bias = bias_variable([1])
    logits = tf.matmul(flattened, weights) + bias

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))

"add batch norm"
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if update_ops:
    updates = tf.group(*update_ops)
    total_loss = control_flow_ops.with_dependencies([updates], cross_entropy)
else:
    total_loss = cross_entropy

optimizer = tf.train.AdamOptimizer(3e-4)
gradients = optimizer.compute_gradients(total_loss)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_step = optimizer.minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Training Loop
step = 0

try:
    while not coord.should_stop():
        start_time = time.time()
        _, loss_value = sess.run([train_step, cross_entropy], feed_dict={keep_prob: 0.75})
        duration = time.time() - start_time

        if step % 100 == 0:
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        step += 1

except tf.errors.OutOfRangeError:
    print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
finally:
    coord.request_stop()

coord.join(threads)
sess.close()
