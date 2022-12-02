#include "sparse.h"
#include "warp.h"
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <eigen3/Eigen/src/Core/Map.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <eigen3/Eigen/src/OrderingMethods/Ordering.h>
#include <eigen3/Eigen/src/SparseCore/MappedSparseMatrix.h>
#include <eigen3/Eigen/src/SparseCore/SparseMatrix.h>
#include <eigen3/Eigen/src/SparseLU/SparseLU.h>

template <typename ValueType = float, typename IndexType = int, int Storage = Eigen::RowMajor>
void _sparse_solve_host(IndexType n, IndexType nnz, IndexType *offsets, IndexType *innerIndex, ValueType *values,
                        ValueType *X, ValueType *Y) {
  Eigen::MappedSparseMatrix<ValueType, Storage, IndexType> A(n, n, nnz, offsets, innerIndex, values);
  Eigen::Map<Eigen::VectorX<ValueType>> b(X, n);
  Eigen::Map<Eigen::VectorX<ValueType>> x(Y, n);

  Eigen::SparseLU<Eigen::SparseMatrix<ValueType, 0, IndexType>, Eigen::COLAMDOrdering<IndexType>> solver;
  solver.analyzePattern(A);
  solver.factorize(A);
  x = solver.solve(b);
}

void csr_solve_host(int n, int nnz, int *offsets, int *columns, float *values, float *X, float *Y) {
  _sparse_solve_host<float, int, Eigen::RowMajor>(n, nnz, offsets, columns, values, X, Y);
}

void csc_solve_host(int n, int nnz, int *offsets, int *rows, float *values, float *X, float *Y) {
  _sparse_solve_host<float, int, Eigen::ColMajor>(n, nnz, offsets, rows, values, X, Y);
}
