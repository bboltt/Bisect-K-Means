import numpy as np

"""
load data as numpy array
"""


def get_dat_sets(file_path):
    samples = np.genfromtxt(file_path)
    return samples

