import os

def get_step_acc(x):
    splitname = x.split(".")
    return tuple([".".join(splitname[:-1])] + [int(splitname[nn].split('_')[-1]) for nn in (0,1)])


def get_prefix(MODEL_DIR, byacc = True):
    "returns: checkpoint file prefix and the step number of the file"
    files = os.listdir(MODEL_DIR)
    step = 0
    acc = 0
    file_= ""
    if byacc:
        for ff, ss, aa in map(get_step_acc , filter(lambda x : x.endswith("index"), files)):
            if (aa > acc) or (aa==acc and ss>step):
                acc = aa
                step = ss
                file_= ff
    else:
        for ff, ss, _ in map(get_step_acc , filter(lambda x : x.endswith("index"), files)):
            if ss>step:
                step = ss
                file_= ff
    return os.path.join(MODEL_DIR, file_), step


#MODEL_DIR = "checkpoints_conv"
#get_prefix(MODEL_DIR)
