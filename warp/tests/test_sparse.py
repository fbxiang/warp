import numpy as np

import warp as wp
from warp.tests.test_base import *

wp.init()

def test_csc_solve_host(test, device):
    n = 4
    nnz = 9
    offsets = wp.array([0, 3, 4, 7, 9], dtype=int, device=device)
    rows = wp.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=int, device=device)
    A = wp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float, device=device)
    '''
        1. 0. 5. 0.
        0. 4. 0. 8.
        2. 0. 6. 0.
        3. 0. 7. 9.
    '''
    X = wp.array([16.0, 40.0, 20.0, 60.0], dtype=float, device=device)
    Y = wp.array([0.0, 0.0, 0.0, 0.0], dtype=float, device=device)
    wp.csc_solve_host(offsets, rows, A, X, Y)
    assert_np_equal(Y.numpy(), np.array([1.0, 2.0, 3.0, 4.0]), tol=1e-4)


def test_csr_solve_host(test, device):
    n = 4
    nnz = 9
    offsets = wp.array([0, 3, 4, 7, 9], dtype=int, device=device)
    cols = wp.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=int, device=device)
    A = wp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float, device=device)
    '''
        1. 0. 2. 3.
        0. 4. 0. 0.
        5. 0. 6. 7.
        0. 8. 0. 9.
    '''
    X = wp.array([19.0, 8.0, 51.0, 52.0], dtype=float, device=device)
    Y = wp.array([0.0, 0.0, 0.0, 0.0], dtype=float, device=device)
    wp.csr_solve_host(offsets, cols, A, X, Y)
    assert_np_equal(Y.numpy(), np.array([1.0, 2.0, 3.0, 4.0]), tol=1e-4)


def test_csr_solve_lt_device(test, device):
    n = 4
    nnz = 9
    offsets = wp.array([0, 3, 4, 7, 9], dtype=int, device=device)
    cols = wp.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=int, device=device)
    A = wp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float, device=device)
    X = wp.array([1.0, 8.0, 23.0, 52.0], dtype=float, device=device)
    Y = wp.array([0.0, 0.0, 0.0, 0.0], dtype=float, device=device)
    wp.csr_solve_lt_device(offsets, cols, A, X, Y)

    assert_np_equal(Y.numpy(), np.array([1.0, 2.0, 3.0, 4.0]), tol=1e-4)


def register(parent):
    devices = wp.get_devices()
    cuda_devices = [d for d in devices if d.is_cuda]
    cpu_devices = [d for d in devices if d.is_cpu]

    class TestSparse(parent):
        pass

    add_function_test(TestSparse, "test_csc_solve_host", test_csc_solve_host, cpu_devices)
    add_function_test(TestSparse, "test_csr_solve_host", test_csr_solve_host, cpu_devices)
    add_function_test(TestSparse, "test_csr_solve_lt_device", test_csr_solve_lt_device, cuda_devices)

    return TestSparse


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
