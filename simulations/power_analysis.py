import random
import sys
import warnings
import numpy as np
from matplotlib import pyplot as plt
from datamatrix import io, operations as ops
import time_series_test as tst


def reduce_signal(dm, signal=.5, subjects=10):
    
    dm = dm.subject_nr == set(random.sample(dm.subject_nr.unique, subjects))
    dm = ops.shuffle(dm)
    for subject_nr, sdm in ops.split(dm.subject_nr):
        chunk_size = int(len(sdm) * (1 - signal))
        sdm = sdm[:chunk_size]
        random.shuffle(sdm.set_size)
        dm.set_size[sdm] = sdm.set_size
    return dm


dm = io.readpickle('data/zhou_et_al_2021.pkl')
N = int(sys.argv[1])
split = int(sys.argv[2])
samples = bool(int(sys.argv[3]))
print('i,split,samples,signal,z')
for i in range(N):
    for signal in np.linspace(0, 1, 6):
        rdm = reduce_signal(dm, signal)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            results = tst.find(
                rdm,
                formula='pupil ~ set_size',
                groups='subject_nr',
                split=split,
                samples_fe=samples,
                samples_re=samples,
                winlen=5)
        print(
            '{},{},{},{},{}'.format(
                i,
                split,
                samples,
                signal,
                results['set_size'].z))
