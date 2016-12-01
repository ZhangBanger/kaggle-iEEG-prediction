import os


def get_step_acc(filename):
    """Get tuple of filename, step index, and accuracy"""
    name_split = filename.split(".")
    return tuple([".".join(name_split[:-1])] + [int(name_split[nn].split('_')[-1]) for nn in (0, 1)])


def get_prefix(model_folder, by_accuracy=False):
    """
    Get checkpoint file prefix and the step number of the file
    By default, returns most recent checkpoint
        If by_accuracy=True, will return checkpoint with highest reported validation accuracy
    """
    files = os.listdir(model_folder)
    step = 0
    acc = 0
    file_ = ""
    if by_accuracy:
        for ff, ss, aa in map(get_step_acc, filter(lambda x: x.endswith("index"), files)):
            if (aa > acc) or (aa == acc and ss > step):
                acc = aa
                step = ss
                file_ = ff
    else:
        for ff, ss, _ in map(get_step_acc, filter(lambda x: x.endswith("index"), files)):
            if ss > step:
                step = ss
                file_ = ff
    return os.path.join(model_folder, file_), step
