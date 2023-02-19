#include "warp.h"
#include <cusparse_v2.h>

namespace wp {

static cusparseHandle_t g_cusparse_handle;


bool init_cusparse() {
  cusparseStatus_t status = cusparseCreate(&g_cusparse_handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "%s\n", cusparseGetErrorString(status));
    return false;
  }
  cusparseSetStream(g_cusparse_handle, (cudaStream_t)cuda_stream_get_current());
  return true;
}
void destroy_cusparse() { cusparseDestroy(g_cusparse_handle); }
void *get_cusparse_handle() { return (void *)g_cusparse_handle; }

} // namespace wp

// -------------------- begin helper functions --------------------

template <typename T> struct CSRMatrix {
  int m{};
  int nnz{};
  int *offsets{};
  int *columns{};
  T *values{};
  cusparseMatDescr_t matDescr{};
  cusparseSpMatDescr_t spMatDescr{};

  csric02Info_t icInfo{};
  csrilu02Info_t iluInfo{};
  cusparseSpSVDescr_t svInfo{};
};

template <typename T> struct DenseVector {
  int m{};
  T *values{};
  cusparseDnVecDescr_t vecDescr{};
};

uint64_t dense_vector_create_device(int m, float *d_values) {
  DenseVector<float> *vec = new DenseVector<float>;
  vec->m = m;
  vec->values = d_values;
  cusparseCreateDnVec(&vec->vecDescr, m, d_values, CUDA_R_32F);
  return (uint64_t)vec;
}

void dense_vector_destroy_device(uint64_t id) {
  DenseVector<void> *vec = (DenseVector<void> *)id;
  cusparseDestroyDnVec(vec->vecDescr);
  delete vec;
}

uint64_t csr_create_device(int m, int nnz, int *d_offsets, int *d_columns, float *d_values, int fillmode,
                           int diagtype) {
  CSRMatrix<float> *mat = new CSRMatrix<float>;
  mat->m = m;
  mat->nnz = nnz;
  mat->offsets = d_offsets;
  mat->columns = d_columns;
  mat->values = d_values;

  cusparseCreateMatDescr(&mat->matDescr);
  cusparseSetMatIndexBase(mat->matDescr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(mat->matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);

  cusparseFillMode_t fillmode_ = (cusparseFillMode_t)fillmode;
  cusparseDiagType_t diagtype_ = (cusparseDiagType_t)diagtype;

  cusparseSetMatFillMode(mat->matDescr, fillmode_);
  cusparseSetMatDiagType(mat->matDescr, diagtype_);

  cusparseCreateCsr(&mat->spMatDescr, m, m, nnz, d_offsets, d_columns, d_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseSpMatSetAttribute(mat->spMatDescr, CUSPARSE_SPMAT_FILL_MODE, &fillmode_, sizeof(cusparseFillMode_t));
  cusparseSpMatSetAttribute(mat->spMatDescr, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype_, sizeof(cusparseDiagType_t));

  return (uint64_t)mat;
}

void csr_destroy_device(uint64_t id) {
  CSRMatrix<void> *mat = (CSRMatrix<void> *)id;
  cusparseDestroyMatDescr(mat->matDescr);
  cusparseDestroySpMat(mat->spMatDescr);
  if (mat->icInfo) {
    cusparseDestroyCsric02Info(mat->icInfo);
  }
  if (mat->iluInfo) {
    cusparseDestroyCsrilu02Info(mat->iluInfo);
  }
  delete mat;
}

// incomplete cholesky
template <typename T>
inline cusparseStatus_t cusparseXcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                                    const cusparseMatDescr_t descrA, T *csrSortedValA,
                                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                    csric02Info_t info, int *pBufferSizeInBytes) {
  if constexpr (std::is_same<T, float>::value) {
    return cusparseScsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                                       pBufferSizeInBytes);
  } else {
    return cusparseDcsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                                       pBufferSizeInBytes);
  }
}

template <typename T>
inline cusparseStatus_t cusparseXcsric02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                  const cusparseMatDescr_t descrA, const T *csrSortedValA,
                                                  const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                  csric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  if constexpr (std::is_same<T, float>::value) {
    return cusparseScsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                                     policy, pBuffer);
  } else {
    return cusparseDcsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                                     policy, pBuffer);
  }
}

template <typename T>
inline cusparseStatus_t cusparseXcsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                         T *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                         const int *csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy,
                                         void *pBuffer) {
  if constexpr (std::is_same<T, float>::value) {
    return cusparseScsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info,
                            policy, pBuffer);
  } else {
    return cusparseDcsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info,
                            policy, pBuffer);
  }
}

// incomplete LU
template <typename T>
inline cusparseStatus_t cusparseXcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                                     const cusparseMatDescr_t descrA, T *csrSortedValA,
                                                     const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                     csrilu02Info_t info, int *pBufferSizeInBytes) {
  if constexpr (std::is_same<T, float>::value) {
    return cusparseScsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                                        pBufferSizeInBytes);
  } else {
    return cusparseDcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                                        pBufferSizeInBytes);
  }
}

template <typename T>
inline cusparseStatus_t cusparseXcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                   const cusparseMatDescr_t descrA, const T *csrSortedValA,
                                                   const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                   csrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  if constexpr (std::is_same<T, float>::value) {
    return cusparseScsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                                      policy, pBuffer);
  } else {
    return cusparseDcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                                      policy, pBuffer);
  }
}

template <typename T>
inline cusparseStatus_t cusparseXcsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                                          T *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                          const int *csrSortedColIndA, csrilu02Info_t info,
                                          cusparseSolvePolicy_t policy, void *pBuffer) {
  if constexpr (std::is_same<T, float>::value) {
    return cusparseScsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info,
                             policy, pBuffer);
  } else {
    return cusparseDcsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info,
                             policy, pBuffer);
  }
}

// -------------------- end helper functions --------------------

template <typename ValueType = float> static int _csr_ichol_device_buffer_size(uint64_t id) {
  cusparseHandle_t cusparse_handle = (cusparseHandle_t)wp::get_cusparse_handle();
  CSRMatrix<ValueType> *mat = (CSRMatrix<ValueType> *)id;
  int bufferSize{0};
  if (!mat->icInfo) {
    cusparseCreateCsric02Info(&mat->icInfo);
  }
  cusparseXcsric02_bufferSize<ValueType>(cusparse_handle, mat->m, mat->nnz, mat->matDescr, mat->values, mat->offsets,
                                         mat->columns, mat->icInfo, &bufferSize);
  return bufferSize;
}

template <typename ValueType = float> static void _csr_ichol_device(uint64_t matA, ValueType *L_values, void *buffer) {
  cusparseHandle_t cusparse_handle = (cusparseHandle_t)wp::get_cusparse_handle();
  CSRMatrix<ValueType> *A = (CSRMatrix<ValueType> *)matA;

  if (!A->icInfo) {
    cusparseCreateCsric02Info(&A->icInfo);
  }

  // L = A
  cudaMemcpyAsync(L_values, A->values, A->nnz * sizeof(ValueType), cudaMemcpyDeviceToDevice);
  cusparseXcsric02_analysis<ValueType>(cusparse_handle, A->m, A->nnz, A->matDescr, L_values, A->offsets, A->columns,
                                       A->icInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer);
  // TODO
  // int structural_zero; cusparseXcsric02_zeroPivot(cusparse_handle, infoM, &structural_zero);
  cusparseXcsric02<ValueType>(cusparse_handle, A->m, A->nnz, A->matDescr, L_values, A->offsets, A->columns, A->icInfo,
                              CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer);
  // TODO
  // int numerical_zero; cusparseXcsric02_zeroPivot(cusparse_handle, infoM, &numerical_zero);
}

template <typename ValueType = float> static int _csr_ilu_device_buffer_size(uint64_t id) {
  cusparseHandle_t cusparse_handle = (cusparseHandle_t)wp::get_cusparse_handle();
  CSRMatrix<ValueType> *mat = (CSRMatrix<ValueType> *)id;
  int bufferSize{0};
  if (!mat->iluInfo) {
    cusparseCreateCsrilu02Info(&mat->iluInfo);
  }
  cusparseXcsrilu02_bufferSize<ValueType>(cusparse_handle, mat->m, mat->nnz, mat->matDescr, mat->values, mat->offsets,
                                          mat->columns, mat->iluInfo, &bufferSize);
  return bufferSize;
}

template <typename ValueType = float> static void _csr_ilu_device(uint64_t matA, ValueType *LU_values, void *buffer) {
  cusparseHandle_t cusparse_handle = (cusparseHandle_t)wp::get_cusparse_handle();
  CSRMatrix<ValueType> *A = (CSRMatrix<ValueType> *)matA;

  if (!A->iluInfo) {
    cusparseCreateCsrilu02Info(&A->iluInfo);
  }

  cudaMemcpyAsync(LU_values, A->values, A->nnz * sizeof(ValueType), cudaMemcpyDeviceToDevice);
  cusparseXcsrilu02_analysis<ValueType>(cusparse_handle, A->m, A->nnz, A->matDescr, LU_values, A->offsets, A->columns,
                                        A->iluInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer);
  // TODO
  // int structural_zero; cusparseXcsrilu02_zeroPivot(cusparse_handle, infoM, &structural_zero);
  cusparseXcsrilu02<ValueType>(cusparse_handle, A->m, A->nnz, A->matDescr, LU_values, A->offsets, A->columns,
                               A->iluInfo, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer);
  // TODO
  // int numerical_zero; cusparseXcsrilu02_zeroPivot(cusparse_handle, infoM, &numerical_zero);
}

template <typename ValueType = float>
static int _csr_mv_device_buffer_size(uint64_t idA, uint64_t idX, uint64_t idY, ValueType alpha, ValueType beta) {
  cusparseHandle_t cusparse_handle = (cusparseHandle_t)wp::get_cusparse_handle();
  CSRMatrix<ValueType> *A = (CSRMatrix<ValueType> *)idA;
  DenseVector<ValueType> *X = (DenseVector<ValueType> *)idX;
  DenseVector<ValueType> *Y = (DenseVector<ValueType> *)idY;
  size_t bufferSize{};
  constexpr cudaDataType valueType = std::is_same<ValueType, float>::value ? CUDA_R_32F : CUDA_R_64F;
  cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A->spMatDescr, X->vecDescr, &beta,
                          Y->vecDescr, valueType, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  return (int)bufferSize;
}

template <typename ValueType = float>
static void _csr_mv_device(uint64_t idA, uint64_t idX, uint64_t idY, ValueType alpha, ValueType beta, void *buffer) {
  cusparseHandle_t cusparse_handle = (cusparseHandle_t)wp::get_cusparse_handle();
  CSRMatrix<ValueType> *A = (CSRMatrix<ValueType> *)idA;
  DenseVector<ValueType> *X = (DenseVector<ValueType> *)idX;
  DenseVector<ValueType> *Y = (DenseVector<ValueType> *)idY;

  constexpr cudaDataType valueType = std::is_same<ValueType, float>::value ? CUDA_R_32F : CUDA_R_64F;
  cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A->spMatDescr, X->vecDescr, &beta,
               Y->vecDescr, valueType, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
}

int csr_ichol_device_buffer_size(uint64_t id) { return _csr_ichol_device_buffer_size<float>(id); }
void csr_ichol_device(uint64_t id, float *L_values, void *buffer) { _csr_ichol_device<float>(id, L_values, buffer); }
int csr_ilu_device_buffer_size(uint64_t id) { return _csr_ilu_device_buffer_size<float>(id); }
void csr_ilu_device(uint64_t id, float *LU_values, void *buffer) { _csr_ilu_device<float>(id, LU_values, buffer); }

int csr_mv_device_buffer_size(uint64_t idA, uint64_t idX, uint64_t idY, float alpha, float beta) {
  return _csr_mv_device_buffer_size<float>(idA, idX, idY, alpha, beta);
}
void csr_mv_device(uint64_t idA, uint64_t idX, uint64_t idY, float alpha, float beta, void *buffer) {
  _csr_mv_device<float>(idA, idX, idY, alpha, beta, buffer);
}

template <typename ValueType>
static int _csr_sv_device_buffer_size(uint64_t idA, uint64_t idX, uint64_t idY, float alpha, int op) {
  cusparseHandle_t handle = (cusparseHandle_t)wp::get_cusparse_handle();
  CSRMatrix<ValueType> *A = (CSRMatrix<ValueType> *)idA;
  DenseVector<ValueType> *X = (DenseVector<ValueType> *)idX;
  DenseVector<ValueType> *Y = (DenseVector<ValueType> *)idY;

  constexpr cudaDataType valueType = std::is_same<ValueType, float>::value ? CUDA_R_32F : CUDA_R_64F;

  if (!A->svInfo) {
    cusparseSpSV_createDescr(&A->svInfo);
  }

  size_t bufferSize{0};
  cusparseSpSV_bufferSize(handle, (cusparseOperation_t)op, &alpha, A->spMatDescr, X->vecDescr, Y->vecDescr, valueType,
                          CUSPARSE_SPSV_ALG_DEFAULT, A->svInfo, &bufferSize);

  return (int)bufferSize;
}

template <typename ValueType>
static void _csr_sv_device(uint64_t idA, uint64_t idX, uint64_t idY, float alpha, int op, void *buffer) {
  cusparseHandle_t handle = (cusparseHandle_t)wp::get_cusparse_handle();
  CSRMatrix<ValueType> *A = (CSRMatrix<ValueType> *)idA;
  DenseVector<ValueType> *X = (DenseVector<ValueType> *)idX;
  DenseVector<ValueType> *Y = (DenseVector<ValueType> *)idY;

  constexpr cudaDataType valueType = std::is_same<ValueType, float>::value ? CUDA_R_32F : CUDA_R_64F;

  if (A->svInfo) {
    cusparseSpSV_createDescr(&A->svInfo);
  }

  cusparseSpSV_analysis(handle, (cusparseOperation_t)op, &alpha, A->spMatDescr, X->vecDescr, Y->vecDescr, valueType,
                        CUSPARSE_SPSV_ALG_DEFAULT, A->svInfo, buffer);
  cusparseSpSV_solve(handle, (cusparseOperation_t)op, &alpha, A->spMatDescr, X->vecDescr, Y->vecDescr, valueType,
                     CUSPARSE_SPSV_ALG_DEFAULT, A->svInfo);
}

int csr_sv_device_buffer_size(uint64_t idA, uint64_t idX, uint64_t idY, float alpha, int op) {
  return _csr_sv_device_buffer_size<float>(idA, idX, idY, alpha, op);
}

void csr_sv_device(uint64_t idA, uint64_t idX, uint64_t idY, float alpha, int op, void *buffer) {
  _csr_sv_device<float>(idA, idX, idY, alpha, op, buffer);
}
