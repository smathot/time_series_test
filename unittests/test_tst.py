import time_series_test as tst
import numpy as np


def _test_indices(fnc):
    
    length = 100
    split = 4
    all_indices = np.arange(length)
    indices = fnc(length, split)
    for test, ref in indices:
        assert(len(test) + len(ref) == len(all_indices))
        assert(set(test) | set(ref) == set(all_indices))
        
        
def test_interleaved_indices():
        
    _test_indices(tst._interleaved_indices)


def test_random_indices():
    
    _test_indices(tst._random_indices)
