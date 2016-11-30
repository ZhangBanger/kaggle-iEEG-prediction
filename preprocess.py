import os

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

from util import subsample, normalize

SUBSAMPLE_RATE = 2
SUBSAMPLE = True
WINDOW_SIZE = 1000
CHANNELS = 16


def mat_to_data(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    for kk, vv in ndata.items():
        if vv.shape == (1, 1):
            ndata[kk] = vv[0, 0]
    return ndata


def get_label(infile):
    return infile.split(".")[-2][-1] == "1"


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def to_example_proto(x, label):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'data': _float_feature(np.hstack(x).astype(dtype=float)),
                'shape': _int64_feature(x.shape),
                'label': _float_feature([label])
            }
        )
    )


def from_example_proto(serialized_example, shape):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'data': tf.FixedLenFeature([shape[0] * shape[1]] , tf.float32),
            'shape': tf.FixedLenFeature([2], tf.int64),
            'label': tf.FixedLenFeature([1], tf.float32),
        }
    )
    x = tf.reshape(features['data'], shape)
    label = features['label']
    return x, label


def write_segments(data_root):
    raw_folder = os.path.join(data_root, "raw")
    file_names = filter(lambda x: x.endswith(".mat"), os.listdir(raw_folder))
    preprocessed_dir = os.path.join(data_root, "preprocessed")

    if not os.path.exists(preprocessed_dir):
        os.mkdir(preprocessed_dir)

    for mat_file_name in file_names:
        train_file_name = os.path.join(preprocessed_dir, mat_file_name.replace(".mat", ".train"))
        valid_file_name = os.path.join(preprocessed_dir, mat_file_name.replace(".mat", ".valid"))
        if os.path.exists(train_file_name) and os.path.exists(valid_file_name):
            print("Skipping existing file:", train_file_name)
            print("Skipping existing file:", valid_file_name)
            continue

        label = get_label(mat_file_name)
        try:
            data = mat_to_data(os.path.join(raw_folder, mat_file_name))
        except ValueError:
            print("Skipping broken file:", mat_file_name)
            continue

        xs = data["data"]
        xs = normalize(xs)

        if SUBSAMPLE:
            xs = subsample(xs, channels=CHANNELS, rate=SUBSAMPLE_RATE)

        num_windows = xs.shape[0] // WINDOW_SIZE
        xs = np.reshape(xs, (num_windows, WINDOW_SIZE, CHANNELS))

        train_writer = tf.python_io.TFRecordWriter(train_file_name)
        valid_writer = tf.python_io.TFRecordWriter(valid_file_name)
        print("Writing file:", train_file_name)
        print("Writing file:", valid_file_name)

        for idx, x in enumerate(xs):
            example = to_example_proto(x, label)
            if idx % 20 == 0:
                valid_writer.write(example.SerializeToString())
            else:
                train_writer.write(example.SerializeToString())

        train_writer.close()
        valid_writer.close()


if __name__ == '__main__':
    data_dir = os.path.expanduser("~/data/seizure-prediction")
    write_segments(data_dir)
