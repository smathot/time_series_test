import time_series_test as tst
import os
import math
import numpy as np
from datamatrix import io, DataMatrix, MultiDimensionalColumn


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


def test_find():
    
    dm = io.readpickle(
        '{}/../data/zhou_et_al_2021.pkl'.format(os.path.dirname(__file__)))
    results = tst.find(dm, 'pupil ~ set_size * color_type',
                       groups='subject_nr', winlen=50)
    tst.summarize(results)
    tst.summarize(results, detailed=True)
    assert math.isclose(results['Intercept'].p, 3.745313085737814e-33)
    assert math.isclose(results['color_type[T.proto]'].p, 0.14493344082757134)
    assert math.isclose(results['set_size'].p, 4.68068027552196e-63)
    assert math.isclose(results['set_size:color_type[T.proto]'].p,
                        0.013513332566898819)


def test_clusters():
    rm = DataMatrix(length=1)
    rm.p = MultiDimensionalColumn(shape=8)
    rm.p = 1, 0, 0, 0, 1, 1, 0, 0
    rm.z = MultiDimensionalColumn(shape=8)
    rm.z = 0, 1, 1, -1, 0, 0, -1, -1
    rm.effect = 'dummy'
    assert tst._clusters(rm, .05) == {
        'dummy': [(1, 3, 2.0), (6, 8, 2.0), (3, 4, 1.0)]}
