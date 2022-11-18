import numpy as np

import warp as wp
from warp.tests.test_base import *

wp.init()


def test_csr_solve_lt_device(test, device):
    n = 4
    nnz = 9
    offsets = wp.array([0, 3, 4, 7, 9], dtype=int, device=device)
    cols = wp.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=int, device=device)
    A = wp.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float, device=device
    )
    X = wp.array([1.0, 8.0, 23.0, 52.0], dtype=float, device=device)
    Y = wp.array([0.0, 0.0, 0.0, 0.0], dtype=float, device=device)
    wp.csr_solve_lt_device(offsets, cols, A, X, Y)

    assert_np_equal(Y.numpy(), np.array([1.0, 2.0, 3.0, 4.0]), tol=1e-4)


def register(parent):
    devices = wp.get_devices()
    devices = [d for d in devices if d.is_cuda]

    class TestSparse(parent):
        pass

    add_function_test(TestSparse, "test_csr_solve_pt", test_csr_solve_lt_device)

    return TestSparse


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
