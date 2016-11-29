import os

import numpy as np
import six
from scipy import signal
from scipy.io import loadmat, savemat


def mat_to_data(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    for kk, vv in six.iteritems(ndata):
        if vv.shape == (1, 1):
            ndata[kk] = vv[0, 0]
    return ndata


def get_label(infile):
    return infile.split(".")[-2][-1] == "0"


def generate_periodograms(folder):
    file_paths = list(filter(lambda x: x.endswith(".mat"), os.listdir(folder)))
    for file_path in file_paths:
        try:
            infile = os.path.join(folder, file_path)
            label = get_label(file_path)
            data = mat_to_data(infile)
            xx = data["data"].transpose(1, 0)
            (freq, powspec) = signal.periodogram(xx)
            yield powspec, np.array([[label]]), "subject" + file_path[:-4]
        except ValueError:
            continue

data_dir = os.path.expanduser("~/data/seizure-prediction")

periodogram_dir = os.path.join(data_dir, "periodograms")

if not os.path.exists(periodogram_dir):
    os.mkdir(periodogram_dir)

for xx, yy, ff in generate_periodograms(data_dir):
    print(ff, xx.shape, yy.shape)
    savemat(os.path.join(periodogram_dir, ff), {"data": xx})
