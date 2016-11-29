import os
from itertools import cycle

import numpy as np
import six
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


def generate_segment(folder):
    file_paths = cycle(filter(lambda x: x.endswith(".mat"), os.listdir(folder)))
    for file_path in file_paths:
        try:
            infile = os.path.join(folder, file_path)
            label = get_label(file_path)
            data = mat_to_data(infile)
            meta = [int(k) for k in file_path.split("/")[-1].split(".")[0].split("_")[:2]]
            xs = data["data"]
            xs = normalize(xs)

            if SUBSAMPLE:
                xs = subsample(xs, channels=CHANNELS, rate=SUBSAMPLE_RATE)

            yield xs, np.array(label), meta
        except ValueError:
            continue


def generate_sample(segment_gen):
    for xs, ys, meta in segment_gen:
        num_windows = xs.shape[0] // WINDOW_SIZE
        xs = np.reshape(xs, (num_windows, WINDOW_SIZE, CHANNELS))
        for x in xs:
            yield x, ys, meta


data_dir = os.path.expanduser("~/data/seizure-prediction")

segment_generator = generate_segment(data_dir)
sample_generator = generate_sample(segment_gen=segment_generator)

if __name__ == '__main__':
    for idx, (x, y, meta) in enumerate(sample_generator):
        if idx % 1000 == 0:
           print("Processing record ", idx)
           print(x, y, meta)