#include "warp.h"
#include <cublas_v2.h>
#include <cusparse_v2.h>

namespace wp {

static cublasHandle_t g_cublas_handle;
static cusparseHandle_t g_cusparse_handle;

bool init_cublas() {
  cublasStatus_t status = cublasCreate(&g_cublas_handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS error: %d\n", status);
    return false;
  }
  return true;
}
void destroy_cublas() { cublasDestroy(g_cublas_handle); }
void *get_cublas_handle() { return (void *)g_cublas_handle; }

bool init_cusparse() {
  cusparseStatus_t status = cusparseCreate(&g_cusparse_handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "%s\n", cusparseGetErrorString(status));
    return false;
  }
  return true;
}
void destroy_cusparse() { cusparseDestroy(g_cusparse_handle); }
void *get_cusparse_handle() { return (void *)g_cusparse_handle; }

} // namespace wp

// -------------------- begin helper functions --------------------

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

template <typename ValueType = float>
static void _csr_ichol_device(int m, int nnz, int *offsets, int *columns, ValueType *values, ValueType *L_values) {
  cudaStream_t stream = (cudaStream_t)cuda_stream_get_current();
  cusparseHandle_t cusparse_handle = (cusparseHandle_t)wp::get_cusparse_handle();
  cublasHandle_t cublas_handle = (cublasHandle_t)wp::get_cublas_handle();
  cusparseSetStream(cusparse_handle, stream);
  cublasSetStream_v2(cublas_handle, stream);

  cusparseMatDescr_t matM{};
  csric02Info_t infoM{};
  int bufferSizeIC{0};
  void *bufferIC;
  cusparseCreateMatDescr(&matM);
  cusparseSetMatIndexBase(matM, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(matM, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(matM, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(matM, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseCreateCsric02Info(&infoM);

  // L = A
  cudaMemcpyAsync(L_values, values, nnz * sizeof(ValueType), cudaMemcpyDeviceToDevice);
  cusparseXcsric02_bufferSize<ValueType>(cusparse_handle, m, nnz, matM, L_values, offsets, columns, infoM,
                                         &bufferSizeIC);
  cudaMallocAsync(&bufferIC, bufferSizeIC, stream); // TODO preallocate
  cusparseXcsric02_analysis<ValueType>(cusparse_handle, m, nnz, matM, L_values, offsets, columns, infoM,
                                       CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferIC);

  // TODO
  // int structural_zero;
  // cusparseXcsric02_zeroPivot(cusparse_handle, infoM, &structural_zero);

  cusparseXcsric02<ValueType>(cusparse_handle, m, nnz, matM, L_values, offsets, columns, infoM,
                              CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferIC);

  // TODO
  // int numerical_zero;
  // cusparseXcsric02_zeroPivot(cusparse_handle, infoM, &numerical_zero);

  cusparseDestroyCsric02Info(infoM);
  cusparseDestroyMatDescr(matM);
  cudaFreeAsync(bufferIC, stream);
}

template <typename ValueType = float>
static void _csr_ilu_device(int m, int nnz, int *offsets, int *columns, ValueType *values, ValueType *LU_values) {
  // L has unit diagonal, U has non-unit diagonal
  cudaStream_t stream = (cudaStream_t)cuda_stream_get_current();
  cusparseHandle_t cusparse_handle = (cusparseHandle_t)wp::get_cusparse_handle();
  cublasHandle_t cublas_handle = (cublasHandle_t)wp::get_cublas_handle();
  cusparseSetStream(cusparse_handle, stream);
  cublasSetStream_v2(cublas_handle, stream);

  cusparseMatDescr_t matM{};
  csrilu02Info_t infoM{};
  int bufferSizeIC{0};
  void *bufferILU;
  cusparseCreateMatDescr(&matM);
  cusparseSetMatIndexBase(matM, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(matM, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(matM, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(matM, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseCreateCsrilu02Info(&infoM);

  // L = A
  cudaMemcpyAsync(LU_values, values, nnz * sizeof(ValueType), cudaMemcpyDeviceToDevice);
  cusparseXcsrilu02_bufferSize<ValueType>(cusparse_handle, m, nnz, matM, LU_values, offsets, columns, infoM,
                                          &bufferSizeIC);
  cudaMallocAsync(&bufferILU, bufferSizeIC, stream); // TODO preallocate
  cusparseXcsrilu02_analysis<ValueType>(cusparse_handle, m, nnz, matM, LU_values, offsets, columns, infoM,
                                        CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferILU);

  // TODO
  // int structural_zero;
  // cusparseXcsric02_zeroPivot(cusparse_handle, infoM, &structural_zero);

  cusparseXcsrilu02<ValueType>(cusparse_handle, m, nnz, matM, LU_values, offsets, columns, infoM,
                               CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferILU);

  // TODO
  // int numerical_zero;
  // cusparseXcsric02_zeroPivot(cusparse_handle, infoM, &numerical_zero);

  cusparseDestroyCsrilu02Info(infoM);
  cusparseDestroyMatDescr(matM);
  cudaFreeAsync(bufferILU, stream);
}

void csr_ichol_device(int m, int nnz, int *offsets, int *columns, float *values, float *L_values) {
  _csr_ichol_device<float>(m, nnz, offsets, columns, values, L_values);
}

void csr_ilu_device(int m, int nnz, int *offsets, int *columns, float *values, float *LU_values) {
  _csr_ilu_device<float>(m, nnz, offsets, columns, values, LU_values);
}

/** Ax = b */
template <typename ValueType = float>
static void csr_ic_cg(int m, int nnz, int *offsets, int *columns, ValueType *values, // A
                      ValueType *p_b,                                                // RHS b
                      ValueType *p_x,                                                // initial x
                      ValueType *p_r,                                                // residual
                      int max_iterations, ValueType rtol) {
  const ValueType minus_one = -1;
  const ValueType one = 1;
  const ValueType zero = 0;
  cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
  cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

  // set up handles
  cudaStream_t stream = (cudaStream_t)cuda_stream_get_current();
  cusparseHandle_t cusparse_handle = (cusparseHandle_t)wp::get_cusparse_handle();
  cublasHandle_t cublas_handle = (cublasHandle_t)wp::get_cublas_handle();
  cusparseSetStream(cusparse_handle, stream);
  cublasSetStream_v2(cublas_handle, stream);

  // set up data type
  static_assert(std::is_same<ValueType, float>::value || std::is_same<ValueType, double>::value,
                "invalid data type for csr_cg");
  constexpr cudaDataType valueType = std::is_same<ValueType, float>::value ? CUDA_R_32F : CUDA_R_64F;

  // set up descriptors
  cusparseSpMatDescr_t matA{};
  cusparseSpMatDescr_t matL{};

  cusparseCreateCsr(&matA, m, m, nnz, offsets, columns, values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, valueType);

  ValueType *L_values{};
  cudaMallocAsync(&L_values, nnz * sizeof(ValueType), stream);
  cudaMemcpyAsync(L_values, values, nnz * sizeof(ValueType), cudaMemcpyDeviceToDevice);

  cusparseCreateCsr(&matL, m, m, nnz, offsets, columns, L_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, valueType);
  cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower));
  cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit, sizeof(diag_non_unit));

  // incomplete Cholesky
  cusparseMatDescr_t matM{};
  csric02Info_t infoM{};
  int bufferSizeIC{0};
  void *bufferIC;
  cusparseCreateMatDescr(&matM);
  cusparseSetMatIndexBase(matM, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(matM, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(matM, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(matM, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseCreateCsric02Info(&infoM);

  if constexpr (std::is_same<ValueType, float>::value) {
    cusparseScsric02_bufferSize(cusparse_handle, m, nnz, matM, L_values, offsets, columns, infoM, &bufferSizeIC);
  } else {
    cusparseDcsric02_bufferSize(cusparse_handle, m, nnz, matM, L_values, offsets, columns, infoM, &bufferSizeIC);
  }
  cudaMallocAsync(&bufferIC, bufferSizeIC, stream); // TODO preallocate

  if constexpr (std::is_same<ValueType, float>::value) {
    cusparseScsric02_analysis(cusparse_handle, m, nnz, matM, L_values, offsets, columns, infoM,
                              CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferIC);
  } else {
    cusparseDcsric02_analysis(cusparse_handle, m, nnz, matM, L_values, offsets, columns, infoM,
                              CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferIC);
  }

  // TODO
  // int structural_zero;
  // cusparseXcsric02_zeroPivot(cusparse_handle, infoM, &structural_zero);

  if constexpr (std::is_same<ValueType, float>::value) {
    cusparseScsric02(cusparse_handle, m, nnz, matM, L_values, offsets, columns, infoM, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                     bufferIC);
  } else {
    cusparseDcsric02(cusparse_handle, m, nnz, matM, L_values, offsets, columns, infoM, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                     bufferIC);
  }

  // TODO
  // int numerical_zero;
  // cusparseXcsric02_zeroPivot(cusparse_handle, infoM, &numerical_zero);

  cusparseDestroyCsric02Info(infoM);
  cusparseDestroyMatDescr(matM);
  cudaFreeAsync(bufferIC, stream);

  //// CG

  // set up
  cusparseDnVecDescr_t vecB{};
  cusparseDnVecDescr_t vecR{};
  cusparseDnVecDescr_t vecX{};
  cusparseDnVecDescr_t vecTmp{};
  cusparseDnVecDescr_t vecRaux{};
  cusparseDnVecDescr_t vecP{};
  cusparseDnVecDescr_t vecT{};

  ValueType *p_tmp{};
  cudaMallocAsync(&p_tmp, m * sizeof(ValueType), stream);

  cusparseCreateDnVec(&vecTmp, m, p_tmp, valueType);
  ValueType *p_raux{};
  cudaMallocAsync(&p_raux, m * sizeof(ValueType), stream);
  cusparseCreateDnVec(&vecRaux, m, p_raux, valueType);

  ValueType *p_p{};
  cudaMallocAsync(&p_p, m * sizeof(ValueType), stream);
  cusparseCreateDnVec(&vecP, m, p_p, valueType);

  ValueType *p_t{};
  cudaMallocAsync(&p_t, m * sizeof(ValueType), stream);
  cusparseCreateDnVec(&vecT, m, p_t, valueType);

  cusparseCreateDnVec(&vecB, m, p_b, valueType);
  cusparseCreateDnVec(&vecR, m, p_r, valueType);
  cusparseCreateDnVec(&vecX, m, p_x, valueType);

  size_t bufferSizeMV{};
  void *bufferMV{};
  // allocate memory
  cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecX, &one, vecB,
                          valueType, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV);
  cudaMallocAsync(&bufferMV, bufferSizeMV, stream); // TODO preallocate and reuse

  // FIXME check bufferSizeMV for all vecs

  // r0 = b - A@x0
  cudaMemcpyAsync(p_r, p_b, m * sizeof(ValueType), cudaMemcpyDeviceToDevice, stream);
  cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecX, &one, vecR, valueType,
               CUSPARSE_SPMV_ALG_DEFAULT, bufferMV);

  // tmp = L^-T r
  size_t bufferSizeL, bufferSizeLT;
  void *bufferL, *bufferLT;
  cusparseSpSVDescr_t spsvDescrL, spsvDescrLT;
  cusparseSpSV_createDescr(&spsvDescrLT);

  cusparseSpSV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL, vecR, vecTmp, valueType,
                          CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, &bufferSizeLT);
  cudaMallocAsync(&bufferLT, bufferSizeLT, stream);
  cusparseSpSV_analysis(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL, vecR, vecTmp, valueType,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, bufferLT);
  cudaMemsetAsync(p_tmp, 0x0, m * sizeof(ValueType), stream);
  cusparseSpSV_solve(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL, vecR, vecTmp, valueType,
                     CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT);

  // raux = L^-1 L^-T r
  cusparseSpSV_createDescr(&spsvDescrL);

  cusparseSpSV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, vecTmp, vecRaux, valueType,
                          CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL);
  cudaMallocAsync(&bufferL, bufferSizeL, stream);
  cusparseSpSV_analysis(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, vecTmp, vecRaux, valueType,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, bufferL);
  cudaMemsetAsync(p_raux, 0x0, m * sizeof(ValueType), stream);
  cusparseSpSV_solve(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, vecTmp, vecRaux, valueType,
                     CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL);

  // p = raux
  cudaMemcpyAsync(p_p, p_raux, m * sizeof(ValueType), cudaMemcpyDeviceToDevice, stream);

  // ValueType r_norm;
  // cublasDnrm2(cublas_handle, m, p_r, 1, &r_norm);

  // delta = r^T r
  ValueType delta;
  ValueType r_norm;
  if constexpr (std::is_same<ValueType, float>::value) {
    cublasSdot_v2(cublas_handle, m, p_r, 1, p_r, 1, &delta);
    cublasSnrm2(cublas_handle, m, p_r, 1, r_norm);
  } else {
    cublasDdot_v2(cublas_handle, m, p_r, 1, p_r, 1, &delta);
    cublasDnrm2(cublas_handle, m, p_r, 1, r_norm);
  }

  ValueType r_norm_init = r_norm;

  for (int i = 0; i < max_iterations; ++i) {
    // t = A p
    cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecP, &zero, vecT, valueType,
                 CUSPARSE_SPMV_ALG_DEFAULT, bufferMV);

    // denom = p^T A p
    ValueType denom;
    if constexpr (std::is_same<ValueType, float>::value) {
      cublasSdot_v2(cublas_handle, m, p_t, 1, p_p, 1, &denom);
    } else {
      cublasDdot_v2(cublas_handle, m, p_t, 1, p_p, 1, &denom);
    }

    // alpha = delta / denom
    ValueType alpha = delta / denom;
    ValueType minus_alpha = -alpha;

    // x = x + alpha * p
    // r = r - alpha * t
    if constexpr (std::is_same<ValueType, float>::value) {
      cublasSaxpy_v2(cublas_handle, m, &alpha, p_p, 1, p_x, 1);
      cublasSaxpy_v2(cublas_handle, m, &minus_alpha, p_t, 1, p_r, 1);
      cublasSnrm2(cublas_handle, m, p_r, 1, r_norm);
    } else {
      cublasDaxpy_v2(cublas_handle, m, &alpha, p_p, 1, p_x, 1);
      cublasDaxpy_v2(cublas_handle, m, &minus_alpha, p_t, 1, p_r, 1);
      cublasDnrm2(cublas_handle, m, p_r, 1, r_norm);
    }

    if (r_norm < r_norm_init * rtol) {
      break;
    }

    cudaMemsetAsync(p_tmp, 0x0, m * sizeof(ValueType), stream);
    cudaMemsetAsync(p_raux, 0x0, m * sizeof(ValueType), stream);

    // tmp = L^-1 r
    cusparseSpSV_solve(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, vecR, vecTmp, valueType,
                       CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL);
    // raux = L^-T L^-1 r
    cusparseSpSV_solve(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL, vecTmp, vecRaux, valueType,
                       CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT);

    ValueType delta_new;
    cublasDdot(cublas_handle, m, p_r, 1, p_r, 1, &delta_new);

    ValueType beta = delta_new / delta;
    delta = delta_new;

    // p = raux
    cudaMemcpyAsync(p_p, p_raux, m * sizeof(ValueType), cudaMemcpyDeviceToDevice, stream);

    // p = beta * p + raux
    cublasDaxpy_v2(cublas_handle, m, &beta, p_p, 1, p_p, 1);
  }

  // r = b
  cudaMemcpyAsync(p_r, p_b, m * sizeof(ValueType), cudaMemcpyDeviceToDevice, stream);

  // r = b - Ax
  cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecX, &one, vecR, valueType,
               CUSPARSE_SPMV_ALG_DEFAULT, bufferMV);

  if constexpr (std::is_same<ValueType, float>::value) {
    cublasSnrm2_v2(cublas_handle, m, p_r, 1, &r_norm);
  } else {
    cublasDnrm2_v2(cublas_handle, m, p_r, 1, &r_norm);
  }
  printf("Final error norm = %e\n", r_norm);

  cusparseSpSV_destroyDescr(spsvDescrLT);
  cusparseSpSV_destroyDescr(spsvDescrL);

  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecR);
  cusparseDestroyDnVec(vecB);
  cusparseDestroyDnVec(vecT);
  cusparseDestroyDnVec(vecP);
  cusparseDestroyDnVec(vecRaux);
  cusparseDestroyDnVec(vecTmp);
  cusparseDestroySpMat(matL);
  cusparseDestroySpMat(matA);

  cudaFreeAsync(bufferL, stream);
  cudaFreeAsync(bufferLT, stream);
  cudaFreeAsync(bufferMV, stream);
  cudaFreeAsync(p_t, stream);
  cudaFreeAsync(p_p, stream);
  cudaFreeAsync(p_raux, stream);
  cudaFreeAsync(p_tmp, stream);
  cudaFreeAsync(L_values, stream);
}

template <cusparseOperation_t op, cusparseFillMode_t fillmode, cusparseDiagType_t diagtype>
static void csr_solve_tri_device_(int n, int nnz, int *offsets, int *columns, float *values, float *X, float *Y) {
  cusparseHandle_t handle = (cusparseHandle_t)wp::get_cusparse_handle();
  cusparseSetStream(handle, (cudaStream_t)cuda_stream_get_current());

  float alpha = 1.f;

  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;

  cusparseSpSVDescr_t spsvDescr;

  cusparseCreateCsr(&matA, n, n, nnz, offsets, columns, values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  cusparseCreateDnVec(&vecX, n, X, CUDA_R_32F);
  cusparseCreateDnVec(&vecY, n, Y, CUDA_R_32F);

  cusparseSpSV_createDescr(&spsvDescr);

  cusparseFillMode_t fillmode_ = fillmode;
  cusparseDiagType_t diagtype_ = diagtype;

  cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE, &fillmode_, sizeof(fillmode));
  cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype_, sizeof(diagtype));

  size_t bufferSize = 0;
  void *buffer = nullptr;
  cusparseSpSV_bufferSize(handle, op, &alpha, matA, vecX, vecY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
                          &bufferSize);

  cudaMalloc(&buffer, bufferSize); // preallocate?
  cusparseSpSV_analysis(handle, op, &alpha, matA, vecX, vecY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, buffer);
  cusparseSpSV_solve(handle, op, &alpha, matA, vecX, vecY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr);

  cusparseDestroySpMat(matA);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseSpSV_destroyDescr(spsvDescr);
  cudaFree(buffer);
}

void csr_solve_lt_device(int n, int nnz, int *offsets, int *columns, float *values, float *X, float *Y) {
  csr_solve_tri_device_<CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT>(
      n, nnz, offsets, columns, values, X, Y);
}
