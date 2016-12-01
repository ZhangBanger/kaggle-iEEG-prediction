#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.python.ops import control_flow_ops

from preprocess import from_example_proto
from util import weight_variable, bias_variable, variable_summaries
from util_tf import get_prefix

"data"
DATA_FOLDER = "~/data/seizure-prediction/preprocessed"

"model saving"
MODEL_DIR = "checkpoints_conv"
REFRESH = False

"hyperparams"
KEEP_PROB = 0.75

NUM_EPOCHS = 10
BATCH_SIZE = 64
EVAL_BATCH = 64
EVAL_EVERY = 2
READ_THREADS = 8
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

keep_prob = tf.placeholder(tf.float32)


def read_and_decode(filename_queue, shape):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    example, label, filename = from_example_proto(serialized_example, shape=shape, filename_queue=filename_queue)

    filename  = filename_queue.dequeue()
    #filename_hash = [int(x) for x in str(filename).split('.')[0].split('_')]
    #if len(filename_hash) ==2:
    #    filename_hash.append(2)
    return example, label, filename


def train_input_pipeline(data_dir, batch_size, read_threads, train=True):
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

def validation_input_pipeline(data_dir, batch_size, read_threads, train=False):
    file_suffix = ".train" if train else ".valid"
    filename_list = list(
        map(
            lambda filename: os.path.join(data_dir, filename),
            filter(lambda filename: filename.endswith(file_suffix), os.listdir(data_dir))
        )
    )
    num_epochs = NUM_EPOCHS if train else None
    #print(*filename_list[:10], sep="\n")
    #print(*filename_list[-10:], sep="\n")
    filename_queue = tf.train.string_input_producer(filename_list, num_epochs=num_epochs)
    shape = (WINDOW_SIZE, CHANNELS)
    example_list = []
    for _ in range(read_threads):
        example_list.append(read_and_decode(filename_queue, shape) )
        #example_list[-1][-1] =  tf.string_to_number()
    print("example_list", len(example_list))

    min_after_dequeue = read_threads * batch_size // 8
    capacity = min_after_dequeue + (read_threads + 2) * batch_size
    #shapes=[shape, [1], [1]],
    return tf.train.batch_join(example_list, batch_size, capacity=32,
              enqueue_many=False,  dynamic_pad=False,
              allow_smaller_final_batch=False, shared_name=None, name=None)



def inference(x):
    with tf.variable_scope("layer1"):
        filter_weights = weight_variable([1, CHANNELS, CHANNELS_L1], name="weights")
        feature_map = tf.nn.conv1d(x, filter_weights, stride=1, padding='SAME')
        feature_map = batch_norm(feature_map, decay=decay_bn, center=True, scale=scale_bn,
                                 epsilon=epsilon_bn, activation_fn=None)
        activation = tf.nn.elu(feature_map + bias_variable(feature_map.get_shape()[1:]))
        activation = tf.nn.dropout(activation, keep_prob=keep_prob)
        activation = tf.reshape(activation, [-1, CHANNELS_L1, WINDOW_SIZE, 1])

    with tf.variable_scope("layer2"):
        filter_weights = weight_variable(KERNEL2, name="weights")
        feature_map = tf.nn.conv2d(activation, filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        feature_map = batch_norm(feature_map, decay=decay_bn, center=True, scale=scale_bn,
                                 epsilon=epsilon_bn, activation_fn=None)

        activation = tf.nn.elu(feature_map + bias_variable(feature_map.get_shape()[1:]))
        activation = tf.nn.max_pool(activation, maxpool_ksize, [1, 1, 1, 1], padding='VALID',
                                    data_format='NHWC', name='maxpool')
        activation = tf.nn.dropout(activation, keep_prob=keep_prob)

    with tf.variable_scope("layer3"):
        filter_weights = weight_variable(KERNEL3, name="weights")
        feature_map = tf.nn.conv2d(activation, filter_weights, strides=[1, 1, 1, 1], padding='SAME')
        feature_map = batch_norm(feature_map, decay=decay_bn, center=True, scale=scale_bn,
                                 epsilon=epsilon_bn, activation_fn=None)

        activation = tf.nn.elu(feature_map + bias_variable(feature_map.get_shape()[1:]))
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
        tf.nn.weighted_cross_entropy_with_logits(logits, y_, pos_weight=3.)
    )

    "add batch norm"
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        return control_flow_ops.with_dependencies([updates], cross_entropy)
    else:
        return cross_entropy


def optimize(loss_op):
    optimizer = tf.train.AdamOptimizer(3e-4)
    grads_and_vars = optimizer.compute_gradients(loss_op)
    for grad, trainable_var in grads_and_vars:
        variable_summaries(grad)
        variable_summaries(trainable_var)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    return global_step, optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)


def evaluation(logits, labels):
    predict_floats = tf.round(tf.nn.sigmoid(logits))
    label_floats = tf.cast(labels, tf.float32)

    raw_accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_floats, label_floats), tf.float32))
    true_positive = predict_floats * label_floats
    precision = tf.reduce_sum(true_positive) / tf.reduce_sum(predict_floats)
    recall = tf.reduce_sum(true_positive) / tf.reduce_sum(label_floats)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return raw_accuracy, precision, recall, f1


DATA_FOLDER = os.path.expanduser(DATA_FOLDER)

def train():
    # Set up training pipeline
    valid_predictors, label_valid = train_input_pipeline(
        data_dir=DATA_FOLDER,
        batch_size=EVAL_BATCH,
        read_threads=READ_THREADS,
        train=False
    )

    batch_predictors, batch_label = train_input_pipeline(
        data_dir=DATA_FOLDER,
        batch_size=BATCH_SIZE,
        read_threads=READ_THREADS,
    )

    batch_logits = inference(batch_predictors)
    batch_loss = loss(batch_logits, batch_label)
    batch_accuracy, batch_precision, batch_recall, batch_f1 = evaluation(batch_logits, batch_label)

    train_step, train_op = optimize(batch_loss)

    # Start graph & runners
    sess = tf.Session()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    saver = tf.train.Saver()
    if not REFRESH:
        checkpoint_file, initial_step = get_prefix(MODEL_DIR, byacc = True)
        print("Restoring the model from a checkpoint:\t%s" % checkpoint_file)
        saver.restore(sess, checkpoint_file)
    else:
        initial_step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("Training Loop")
    step = 0
    try:
        while not coord.should_stop():
            start_time = time.time()
            _, step, train_loss, train_acc, train_prec, train_rec, train_f1 = sess.run(
                [train_op, train_step, batch_loss,
                 batch_accuracy, batch_precision, batch_recall, batch_f1],
                feed_dict={keep_prob: KEEP_PROB}
            )
            duration = time.time() - start_time

            if step % EVAL_EVERY == 0:
                valid_xs, valid_ys = sess.run([valid_predictors, label_valid])
                valid_loss, valid_acc, valid_prec, valid_rec, valid_f1 = sess.run(
                    [batch_loss, batch_accuracy, batch_precision, batch_recall, batch_f1],
                    feed_dict={batch_predictors: valid_xs, batch_label: valid_ys, keep_prob: 1.}
                    )
                chkpt_file = MODEL_DIR + "/" + "step_%u.val_acc_%u" % \
                                    (initial_step+step, int(1000*valid_acc))
                save_path = saver.save(sess, chkpt_file)
                print('Step %d (with previous: %d) (%3f sec)' % (step, initial_step+step, duration))
                print('train-loss = %.2f, train-acc = %.3f, train-prec = %.2f, train-rec = %.2f, train-f1 = %.2f' % (
                    train_loss, train_acc, train_prec, train_rec, train_f1
                ))
                print('valid-loss = %.2f, valid-acc = %.3f, valid-prec = %.2f, valid-rec = %.2f, valid-f1 = %.2f' % (
                    valid_loss, valid_acc, valid_prec, valid_rec, valid_f1
                ))

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    return

def get_file_name_for_table(ff, mat = True):
    ff = ff.decode("ascii").split("/")[-1].split(".")[0]
    return ff + ".mat"

def predict(outfile, SEP="\t", mode = "w+"):

    valid_predictors, label_valid, file_valid = validation_input_pipeline(
        data_dir=DATA_FOLDER,
        batch_size=EVAL_BATCH,
        read_threads=READ_THREADS,
        train=False
    )
    batch_logits = inference(valid_predictors)
    # Start graph & runners
    sess = tf.Session()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    saver = tf.train.Saver()
    checkpoint_file, initial_step = get_prefix(MODEL_DIR, byacc = True)
    print("Restoring the model from a checkpoint:\t%s" % checkpoint_file)
    saver.restore(sess, checkpoint_file)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("Predicting")
    with open(outfile, mode=mode) as outfilehandle:
        print( "prediction_file", "predicted_label", file=outfilehandle, sep=SEP)
        step = 0
        try:
            while not coord.should_stop():
                step +=1
                start_time = time.time()
                prediction_file, predicted_label = sess.run([ file_valid, batch_logits],
                    feed_dict={ keep_prob: 1.}
                                                           )
                duration = time.time() - start_time
                print("prediction batch %u (%.2f s)" %(step, duration))
                for ff, logit_ in zip(prediction_file.ravel(), predicted_label.ravel()):
                    probability = 1/(1+np.exp(-logit_))
                    print(get_file_name_for_table(ff), "%.6f" % probability,
                          file=outfilehandle, sep=SEP)

        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()
    return

if __name__ == "__main__":
    if (len(sys.argv)>1) and (sys.argv[1] in ("p", "predict")):
        print("prediction")
        outfile = "prediction.csv"
        predict(outfile, mode="w+")
    else:
        print("training")
        train()
