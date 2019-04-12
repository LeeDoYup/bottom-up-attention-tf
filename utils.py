from __future__ import print_function

import os
import numpy as np

EPS = 1e-7

def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def convert_entries(entries):
    new_entries = {}
    entry_keys = list(entries[0].keys())
    for key in entry_keys:
        temp = [entry[key] for entry in entries]
        new_entries[key] = np.array(temp)
    return new_entries


def get_h5py_path(dataroot, name):
    assert name in ['train', 'val']
    h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
    return h5_path


if __name__ == '__main__':
    entry = {'hi': 123, 'test': 456}
    entries = [entry, entry, entry]
    test = convert_entries(entries)
    assert type(test) == type(entry)
    assert list(test.keys()) == list(entry.keys())
    print(test)