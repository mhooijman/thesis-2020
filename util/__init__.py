import os
import errno
import numpy as np


def _try_to_create_dir(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def write_to_file(path, content, mode):
    _try_to_create_dir(path)

    with open(path, mode) as f:
        f.write(content)

def write_numpy_to_file(path, array):
    _try_to_create_dir(path)
    np.save(path, array)