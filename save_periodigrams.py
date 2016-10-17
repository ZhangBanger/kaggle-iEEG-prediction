import os
import numpy as np
from scipy import signal
from scipy.io import loadmat
import six

def mat_to_data(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    for kk,vv in six.iteritems(ndata):
#         print(vv.shape)
        if vv.shape == (1,1):
            ndata[kk] = vv[0,0]
    return ndata

def get_label(infile):
    return infile.split(".")[-2][-1] == "0"

def periodigram_gen_one_name(folder):
    infiles = list(filter(lambda x: x.endswith(".mat"), os.listdir(folder)))
    NUM_FILES = len(infiles)
    for nn, ff in enumerate(infiles):
        infile = os.path.join(folder, ff)
        label = get_label(ff)
        data = mat_to_data(infile)
        xx = data["data"].transpose(1,0)
        (freq, powspec) = signal.periodogram(xx)
        yield powspec, np.array([[label]]), ff[:-4]

datadir = "data/"

from pandas import HDFStore, DataFrame
h5filename = os.path.join( datadir, "periodigrams.h5")

with HDFStore(h5filename) as h5:
    for xx, yy, ff in periodigram_gen_one_name(datadir):
        print(ff, xx.shape, yy.shape)
        dfx = DataFrame(xx)
        dfx.name = yy
        h5[ff] = DataFrame(xx)


h5filename = "data/periodigrams.h5"
h5 = HDFStore(h5filename)
print("=====================")
print("keys:")
print(h5.keys())
h5.close()
