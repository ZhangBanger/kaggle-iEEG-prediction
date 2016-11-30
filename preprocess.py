import os

import numpy as np
import six
import tensorflow as tf
from scipy.io import loadmat

from util import subsample, normalize

SUBSAMPLE_RATE = 2
SUBSAMPLE = True
WINDOW_SIZE = 1000
CHANNELS = 16
TABLE_NAME = "train"


def mat_to_data(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    for kk, vv in six.iteritems(ndata):
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


def write_segments(data_root):
    file_names = filter(lambda x: x.endswith(".mat"), os.listdir(data_root))
    segment_dir = os.path.join(data_root, "segments")

    if not os.path.exists(segment_dir):
        os.mkdir(segment_dir)

    for mat_file_name in file_names:
        segment_file_name = os.path.join(segment_dir, mat_file_name.replace(".mat", ".tfrecords"))
        label = get_label(mat_file_name)
        data = mat_to_data(os.path.join(data_root, mat_file_name))

        xs = data["data"]
        xs = normalize(xs)

        if SUBSAMPLE:
            xs = subsample(xs, channels=CHANNELS, rate=SUBSAMPLE_RATE)

        num_windows = xs.shape[0] // WINDOW_SIZE
        xs = np.reshape(xs, (num_windows, WINDOW_SIZE, CHANNELS))

        writer = tf.python_io.TFRecordWriter(segment_file_name)
        print("Writing file:", segment_file_name)

        for x in xs:
            example = to_example_proto(x, label)
            writer.write(example.SerializeToString())

        writer.close()

if __name__ == '__main__':
    data_dir = os.path.expanduser("~/data/seizure-prediction")
    write_segments(data_dir)
