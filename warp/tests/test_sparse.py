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
    """
        1. 0. 5. 0.
        0. 4. 0. 8.
        2. 0. 6. 0.
        3. 0. 7. 9.
    """
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
    """
        1. 0. 2. 3.
        0. 4. 0. 0.
        5. 0. 6. 7.
        0. 8. 0. 9.
    """
    X = wp.array([19.0, 8.0, 51.0, 52.0], dtype=float, device=device)
    Y = wp.array([0.0, 0.0, 0.0, 0.0], dtype=float, device=device)
    wp.csr_solve_host(offsets, cols, A, X, Y)
    assert_np_equal(Y.numpy(), np.array([1.0, 2.0, 3.0, 4.0]), tol=1e-4)


def test_csr_pd_solve_host(test, device):
    n = 4
    nnz = 9
    offsets = wp.array([0, 3, 4, 7, 10], dtype=int, device=device)
    cols = wp.array([0, 2, 3, 1, 0, 2, 3, 0, 2, 3], dtype=int, device=device)
    A = wp.array([10.0, 2.0, 3.0, 4.0, 2.0, 6.0, 7.0, 3.0, 7.0, 9.0], dtype=float, device=device)

    """
        10. 0. 2. 3.
        0.  4. 0. 0.
        2.  0. 6. 7.
        3.  0. 7. 9.
    """

    X = wp.array([28.0, 8.0, 48.0, 60.0], dtype=float, device=device)
    Y = wp.array([0.0, 0.0, 0.0, 0.0], dtype=float, device=device)
    wp.csr_pd_solve_host(offsets, cols, A, X, Y)
    assert_np_equal(Y.numpy(), np.array([1.0, 2.0, 3.0, 4.0]), tol=1e-4)

    # Only LT is needed
    offsets = wp.array([0, 1, 2, 4, 7], dtype=int, device=device)
    cols = wp.array([0, 1, 0, 2, 0, 2, 3], dtype=int, device=device)
    A = wp.array([10.0, 4.0, 2.0, 6.0, 3.0, 7.0, 9.0], dtype=float, device=device)

    """
        10. 0. 0. 0.
        0.  4. 0. 0.
        2.  0. 6. 0.
        3.  0. 7. 9.
    """

    X = wp.array([28.0, 8.0, 48.0, 60.0], dtype=float, device=device)
    Y = wp.array([0.0, 0.0, 0.0, 0.0], dtype=float, device=device)
    wp.csr_pd_solve_host(offsets, cols, A, X, Y)
    assert_np_equal(Y.numpy(), np.array([1.0, 2.0, 3.0, 4.0]), tol=1e-4)



# def test_csr_ilu_device(test, device):
#     n = 4
#     nnz = 9
#     offsets = wp.array([0, 3, 4, 7, 9], dtype=int, device=device)
#     cols = wp.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=int, device=device)
#     A = wp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float, device=device)
#     LU = wp.zeros_like(A)

#     buffer_size = wp.csr_ilu_device_buffer_size(offsets, cols, A)
#     buffer = wp.array(np.zeros(buffer_size, dtype=np.int8), dtype=wp.int8, device=device)

#     wp.csr_ilu_device(offsets, cols, A, LU, buffer)
#     # L = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [5, 0, 1, 0], [0, 2, 0, 1]], dtype=np.float32)
#     # U = np.array([[1, 0, 2, 3], [0, 4, 0, 0], [0, 0, -4, -8], [0, 0, 0, 9]], dtype=np.float32)

#     wp.synchronize()
#     assert_np_equal(
#         LU.numpy(),
#         np.array([1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -8.0, 2.0, 9.0]),
#         tol=1e-4,
#     )


def test_csr_ilu_device(test, device):
    n = 4
    nnz = 9
    offsets = wp.array([0, 3, 4, 7, 9], dtype=int, device=device)
    cols = wp.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=int, device=device)
    A = wp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float, device=device)
    LU = wp.zeros_like(A)

    mat = wp.types.CSRMatrix(offsets, cols, A)
    buffer_size = wp.context.runtime.core.csr_ilu_device_buffer_size(mat.id)
    buffer = wp.array(np.zeros(buffer_size, dtype=np.int8), dtype=wp.int8, device=device)
    wp.context.runtime.core.csr_ilu_device(mat.id, LU.ptr, buffer.ptr)
    wp.synchronize()
    assert_np_equal(
        LU.numpy(),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -8.0, 2.0, 9.0]),
        tol=1e-4,
    )


def test_csr_ichol_device(test, device):
    n = 4
    nnz = 9
    offsets = wp.array([0, 2, 4, 6, 8], dtype=int, device=device)
    cols = wp.array([0, 2, 1, 3, 0, 2, 1, 3], dtype=int, device=device)
    A = wp.array([1, 5, 1, 2, 5, 26, 2, 5], dtype=float, device=device)
    L = wp.zeros_like(A)

    mat = wp.types.CSRMatrix(offsets, cols, A)
    buffer_size = wp.context.runtime.core.csr_ichol_device_buffer_size(mat.id)
    buffer = wp.array(np.zeros(buffer_size, dtype=np.int8), dtype=wp.int8, device=device)
    wp.context.runtime.core.csr_ichol_device(mat.id, L.ptr, buffer.ptr)
    wp.synchronize()
    assert_np_equal(
        L.numpy()[[0, 2, 4, 5, 6, 7]],
        np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0]),
        tol=1e-4,
    )


def test_csr_mv_device(test, device):
    n = 4
    nnz = 9
    offsets = wp.array([0, 3, 4, 7, 9], dtype=int, device=device)
    cols = wp.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=int, device=device)
    A = wp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float, device=device)
    x = wp.array([1, 0, 1, 0], dtype=float, device=device)
    y = wp.array([0, 0, 0, 0], dtype=float, device=device)
    mat = wp.types.CSRMatrix(offsets, cols, A)
    vec_x = wp.types.DenseVector(x)
    vec_y = wp.types.DenseVector(y)

    buffer_size = wp.context.runtime.core.csr_mv_device_buffer_size(mat.id, vec_x.id, vec_y.id, 1, 1)
    buffer = wp.array(np.zeros(buffer_size, dtype=np.int8), dtype=wp.int8, device=device)
    wp.context.runtime.core.csr_mv_device(mat.id, vec_x.id, vec_y.id, 1, 1, buffer.ptr)
    wp.synchronize()
    assert_np_equal(y.numpy(), np.array([3.0, 0.0, 11.0, 0.0]), tol=1e-4)


def test_csr_sv_device(test, device):
    NON_TRANSPOSE = 0
    TRANSPOSE = 1

    n = 4
    nnz = 9
    offsets = wp.array([0, 3, 4, 7, 9], dtype=int, device=device)
    cols = wp.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=int, device=device)
    A = wp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float, device=device)
    X = wp.array([1.0, 8.0, 23.0, 52.0], dtype=float, device=device)
    Y = wp.array([0.0, 0.0, 0.0, 0.0], dtype=float, device=device)
    vec_x = wp.types.DenseVector(X)
    vec_y = wp.types.DenseVector(Y)

    mat = wp.types.CSRMatrix(offsets, cols, A, fill_mode="lower", diag_type="non_unit")

    buffer_size = wp.context.runtime.core.csr_sv_device_buffer_size(mat.id, vec_x.id, vec_y.id, 1, NON_TRANSPOSE)
    buffer = wp.array(np.zeros(buffer_size, dtype=np.int8), dtype=wp.int8, device=device)
    wp.context.runtime.core.csr_sv_device(mat.id, vec_x.id, vec_y.id, 1, NON_TRANSPOSE, buffer.ptr)
    wp.synchronize()
    assert_np_equal(Y.numpy(), np.array([1.0, 2.0, 3.0, 4.0]), tol=1e-4)

    X.assign([16.0, 40.0, 18.0, 36.0])
    buffer_size = wp.context.runtime.core.csr_sv_device_buffer_size(mat.id, vec_x.id, vec_y.id, 1, TRANSPOSE)
    buffer = wp.array(np.zeros(buffer_size, dtype=np.int8), dtype=wp.int8, device=device)
    wp.context.runtime.core.csr_sv_device(mat.id, vec_x.id, vec_y.id, 1, TRANSPOSE, buffer.ptr)
    wp.synchronize()
    assert_np_equal(Y.numpy(), np.array([1.0, 2.0, 3.0, 4.0]), tol=1e-4)

    mat = wp.types.CSRMatrix(offsets, cols, A, fill_mode="upper", diag_type="non_unit")

    X.assign([19.0, 8.0, 46.0, 36.0])
    buffer_size = wp.context.runtime.core.csr_sv_device_buffer_size(mat.id, vec_x.id, vec_y.id, 1, NON_TRANSPOSE)
    buffer = wp.array(np.zeros(buffer_size, dtype=np.int8), dtype=wp.int8, device=device)
    wp.context.runtime.core.csr_sv_device(mat.id, vec_x.id, vec_y.id, 1, NON_TRANSPOSE, buffer.ptr)
    wp.synchronize()
    assert_np_equal(Y.numpy(), np.array([1.0, 2.0, 3.0, 4.0]), tol=1e-4)

    X.assign([1.0, 8.0, 20.0, 60.0])
    buffer_size = wp.context.runtime.core.csr_sv_device_buffer_size(mat.id, vec_x.id, vec_y.id, 1, TRANSPOSE)
    buffer = wp.array(np.zeros(buffer_size, dtype=np.int8), dtype=wp.int8, device=device)
    wp.context.runtime.core.csr_sv_device(mat.id, vec_x.id, vec_y.id, 1, TRANSPOSE, buffer.ptr)
    wp.synchronize()
    assert_np_equal(Y.numpy(), np.array([1.0, 2.0, 3.0, 4.0]), tol=1e-4)

    mat = wp.types.CSRMatrix(offsets, cols, A, fill_mode="lower", diag_type="unit")

    X.assign([1.0, 2.0, 8.0, 20.0])
    buffer_size = wp.context.runtime.core.csr_sv_device_buffer_size(mat.id, vec_x.id, vec_y.id, 1, NON_TRANSPOSE)
    buffer = wp.array(np.zeros(buffer_size, dtype=np.int8), dtype=wp.int8, device=device)
    wp.context.runtime.core.csr_sv_device(mat.id, vec_x.id, vec_y.id, 1, NON_TRANSPOSE, buffer.ptr)
    wp.synchronize()
    assert_np_equal(Y.numpy(), np.array([1.0, 2.0, 3.0, 4.0]), tol=1e-4)


def register(parent):
    devices = wp.get_devices()
    cuda_devices = [d for d in devices if d.is_cuda]
    cpu_devices = [d for d in devices if d.is_cpu]

    class TestSparse(parent):
        pass

    add_function_test(TestSparse, "test_csc_solve_host", test_csc_solve_host, cpu_devices)
    add_function_test(TestSparse, "test_csr_solve_host", test_csr_solve_host, cpu_devices)
    add_function_test(TestSparse, "test_csr_pd_solve_host", test_csr_pd_solve_host, cpu_devices)
    add_function_test(TestSparse, "test_csr_sv_device", test_csr_sv_device, cuda_devices)
    add_function_test(TestSparse, "test_csr_ilu_device", test_csr_ilu_device, cuda_devices)
    add_function_test(TestSparse, "test_csr_mv_device", test_csr_mv_device, cuda_devices)
    add_function_test(TestSparse, "test_csr_ichol_device", test_csr_ichol_device, cuda_devices)

    return TestSparse


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
