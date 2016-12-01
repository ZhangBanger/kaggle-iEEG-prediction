#!/usr/bin/env python3
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training.saver import latest_checkpoint

from preprocess import from_example_proto, generate_test_segment, PREPROCESSED_DIR
from util import weight_variable, bias_variable, variable_summaries

# Directory Structure
RUN_ID = "eeg-conv-pos-weight-1"
DATA_ROOT = os.path.expanduser("~/data/seizure-prediction")
LOG_DIR = os.path.join(DATA_ROOT, "log", RUN_ID)
MODEL_DIR = os.path.join(DATA_ROOT, "model", RUN_ID)
OUTPUT_DIR = os.path.join(DATA_ROOT, "output", RUN_ID)
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# General HyperParameters
KEEP_PROB = 0.75
LEARNING_RATE = 3e-4
LR_DECAY = 0.9
LR_DECAY_STEPS = 1000
NUM_EPOCHS = 10
BATCH_SIZE = 256
EVAL_BATCH = 1024
EVAL_EVERY = 100
READ_THREADS = 8
WINDOW_SIZE = 1000
POSITIVE_WEIGHT = 1.

# Convolutional HyperParameters
CHANNELS = 16
CHANNELS_L1 = 32
CHANNELS_L2 = 4
CHANNELS_L3 = 2
maxpool_ksize = [1, 2, 4, 1]
KERNEL2 = [32, 2, 1, CHANNELS_L2]
KERNEL3 = [8, 4, CHANNELS_L2, CHANNELS_L3]
scale_bn = False
decay_bn = 0.999
epsilon_bn = 0.001

keep_prob = tf.placeholder(tf.float32)


def read_and_decode(filename_queue, shape):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    example, label = from_example_proto(serialized_example, shape=shape)

    return example, label


def input_pipeline(data_dir, batch_size, read_threads, train=True):
    file_suffix = ".train" if train else ".valid"
    filename_list = list(
        map(
            lambda filename: os.path.join(data_dir, filename),
            filter(lambda filename: filename.endswith(file_suffix), os.listdir(data_dir))
        )
    )
    num_epochs = NUM_EPOCHS if train else None
    filename_queue = tf.train.string_input_producer(filename_list, num_epochs=num_epochs)
    shape = (WINDOW_SIZE, CHANNELS)
    example_list = [read_and_decode(filename_queue, shape)[:2] for _ in range(read_threads)]

    min_after_dequeue = read_threads * batch_size // 8
    capacity = min_after_dequeue + (read_threads + 2) * batch_size
    return tf.train.shuffle_batch_join(
        example_list,
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        allow_smaller_final_batch=True,
    )


def inference(x, is_training=True):
    with tf.variable_scope("layer1"):
        filter_weights = weight_variable([1, CHANNELS, CHANNELS_L1], name="weights")
        feature_map = tf.nn.conv1d(x, filter_weights, stride=1, padding='SAME')
        feature_map = batch_norm(feature_map, decay=decay_bn, center=True, scale=scale_bn,
                                 epsilon=epsilon_bn, activation_fn=None, is_training=is_training)
        activation = tf.nn.elu(feature_map)
        activation = tf.nn.dropout(activation, keep_prob=keep_prob)
        activation = tf.reshape(activation, [-1, CHANNELS_L1, WINDOW_SIZE, 1])

    with tf.variable_scope("layer2"):
        filter_weights = weight_variable(KERNEL2, name="weights")
        feature_map = tf.nn.conv2d(activation, filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        feature_map = batch_norm(feature_map, decay=decay_bn, center=True, scale=scale_bn,
                                 epsilon=epsilon_bn, activation_fn=None, is_training=is_training)

        activation = tf.nn.elu(feature_map)
        activation = tf.nn.max_pool(activation, maxpool_ksize, [1, 1, 1, 1], padding='VALID',
                                    data_format='NHWC', name='maxpool')
        activation = tf.nn.dropout(activation, keep_prob=keep_prob)

    with tf.variable_scope("layer3"):
        filter_weights = weight_variable(KERNEL3, name="weights")
        feature_map = tf.nn.conv2d(activation, filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        feature_map = batch_norm(feature_map, decay=decay_bn, center=True, scale=scale_bn,
                                 epsilon=epsilon_bn, activation_fn=None, is_training=is_training)

        activation = tf.nn.elu(feature_map)
        activation = tf.nn.max_pool(activation, maxpool_ksize, [1, 1, 1, 1], padding='VALID',
                                    data_format='NHWC', name='maxpool')
        activation = tf.nn.dropout(activation, keep_prob=keep_prob)

    with tf.variable_scope("layer4"):
        dim = np.prod(activation.get_shape().as_list()[1:])
        flattened = tf.reshape(activation, [-1, dim])
        weights = weight_variable([dim, 1])
        bias = bias_variable([1])
        logits = tf.matmul(flattened, weights) + bias

    return logits


def loss(logits, y_):
    cross_entropy = tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(logits, y_, pos_weight=POSITIVE_WEIGHT)
    )
    tf.scalar_summary("loss", cross_entropy)

    # Include batch norm as dependency so parameters can update
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        return control_flow_ops.with_dependencies([updates], cross_entropy)
    else:
        return cross_entropy


def optimize(loss_op):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                               LR_DECAY_STEPS, LR_DECAY, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss_op)
    for grad, trainable_var in grads_and_vars:
        variable_summaries(grad)
        variable_summaries(trainable_var)
    return global_step, optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)


def evaluation(logits, labels):
    predict_floats = tf.round(tf.nn.sigmoid(logits), name="predictions")
    variable_summaries(predict_floats)
    label_floats = tf.cast(labels, tf.float32)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_floats, label_floats), tf.float32))
    tf.scalar_summary("accuracy", accuracy)
    auc, update_auc = tf.contrib.metrics.streaming_auc(predict_floats, label_floats)
    tf.scalar_summary("auc", auc)

    return accuracy, auc, update_auc


def train_model():
    input_folder = os.path.join(DATA_ROOT, PREPROCESSED_DIR)
    # Set up training pipeline
    valid_predictors, valid_label = input_pipeline(
        data_dir=input_folder,
        batch_size=EVAL_BATCH,
        read_threads=READ_THREADS,
        train=False
    )

    train_predictors, train_label = input_pipeline(
        data_dir=input_folder,
        batch_size=BATCH_SIZE,
        read_threads=READ_THREADS,
    )

    batch_logits = inference(train_predictors)
    batch_loss = loss(batch_logits, train_label)
    batch_accuracy, batch_auc, update_auc = evaluation(batch_logits, train_label)

    train_step, train_op = optimize(batch_loss)

    # Start graph & runners
    sess = tf.Session()
    merged = tf.summary.merge_all()
    train_writer = tf.train.SummaryWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
    valid_writer = tf.train.SummaryWriter(os.path.join(LOG_DIR, 'test'))

    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    if latest_checkpoint(MODEL_DIR):
        checkpoint_file = latest_checkpoint(MODEL_DIR)
        print("Restoring the model from most recent checkpoint:\t%s" % checkpoint_file)
        saver.restore(sess, checkpoint_file)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("Training Loop")
    step = 0
    try:
        while not coord.should_stop():
            start_time = time.time()
            _, train_summary, step, train_loss, train_acc, train_auc, _ = sess.run(
                [train_op, merged, train_step, batch_loss, batch_accuracy, batch_auc, update_auc],
                feed_dict={keep_prob: KEEP_PROB}
            )
            train_writer.add_summary(train_summary, step)
            duration = time.time() - start_time

            if step % EVAL_EVERY == 0:
                valid_xs, valid_ys = sess.run([valid_predictors, valid_label])
                valid_summary, valid_loss, valid_acc, valid_auc, _ = sess.run(
                    [merged, batch_loss, batch_accuracy, batch_auc, update_auc],
                    feed_dict={train_predictors: valid_xs, train_label: valid_ys, keep_prob: 1.}
                )
                valid_writer.add_summary(valid_summary, step)
                checkpoint_file = os.path.join(
                    MODEL_DIR,
                    "val_auc_%u" % int(1000 * valid_auc)
                )
                saver.save(sess, checkpoint_file, global_step=step)
                print('Step %d (%3f sec)' % (step, duration))
                print('train-loss = %.2f, train-acc = %.3f, train-auc = %.2f' % (
                    train_loss, train_acc, train_auc
                ))
                print('valid-loss = %.2f, valid-acc = %.3f, valid-auc = %.2f' % (
                    valid_loss, valid_acc, valid_auc
                ))

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    return


def predict(output_path, separator=",", mode="w+"):
    print("Setting up inference subgraph")
    predict_input = tf.placeholder(dtype=tf.float32, shape=[None, WINDOW_SIZE, CHANNELS])
    batch_logits = inference(predict_input, is_training=False)
    predicted_probabilities = tf.nn.sigmoid(batch_logits)
    mean_prediction = tf.reduce_mean(predicted_probabilities)

    print("Restoring model from training with best validation accuracy")
    sess = tf.Session()
    saver = tf.train.Saver()
    checkpoint_file = latest_checkpoint(MODEL_DIR)
    print("Restoring the model from a checkpoint:\t%s" % checkpoint_file)
    saver.restore(sess, checkpoint_file)

    print("Predicting")
    with open(output_path, mode=mode) as file_stream:
        print("File", "Class", file=file_stream, sep=separator)
        for segment, file_name in generate_test_segment(DATA_ROOT, "test"):
            predicted_probability = sess.run(mean_prediction, feed_dict={predict_input: segment, keep_prob: 1.})
            print(file_name, predicted_probability, sep=separator, file=file_stream)


if __name__ == "__main__":
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_bool('predict', False, 'Run prediction or train [default]')
    if FLAGS.predict:
        output_file = "prediction.csv"
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)

        predict(os.path.join(OUTPUT_DIR, output_file), mode="w+")
    else:
        print("training")
        train_model()
