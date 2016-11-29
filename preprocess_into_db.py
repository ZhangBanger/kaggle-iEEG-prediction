import os
import sqlite3

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
    return infile.split(".")[-2][-1] == "0"


def generate_segment(folder):
    file_paths = list(filter(lambda x: x.endswith(".mat"), os.listdir(folder)))
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

# Time to save to DB!
db_name = "%s_piece_%u.db" % (TABLE_NAME, WINDOW_SIZE)
db_path = os.path.join(data_dir, db_name)

print("Saving DB to ", db_path)

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()

    print("PURGING")
    drop_table_query = "DROP TABLE IF EXISTS %s" % TABLE_NAME
    cursor.execute(drop_table_query)

    print("INITIALIZING")
    create_table_query = """CREATE TABLE IF NOT EXISTS %s (
            id INT PRIMARY KEY,
            label INT,
            data BLOB,
            individual INT,
            segment INT
            )""" % TABLE_NAME
    cursor.execute(create_table_query)

    insert_query = "INSERT INTO %s (id, label, data, individual, segment) VALUES (?,?,?,?,?)" % TABLE_NAME

    print("INSERTING SAMPLES")
    for idx, (x, y, meta) in enumerate(sample_generator):
        label = bool(y)
        blob = sqlite3.Binary(x.tobytes())
        cursor.execute(insert_query, (idx, label, blob, meta[0], meta[1]))
        if idx % 100 == 0:
            print("Finished %i samples, currently on %s - %s" % (idx, meta[0], meta[1]))
