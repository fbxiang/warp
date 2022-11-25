#include "sparse.h"
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
  return true;
}
void destroy_cusparse() { cusparseDestroy(g_cusparse_handle); }

void* get_cusparse_handle() { return (void*)g_cusparse_handle; }

} // namespace wp

template <cusparseFillMode_t fillmode, cusparseDiagType_t diagtype>
static void csr_solve_device_(int n, int nnz, int *offsets, int *columns,
                              float *values, float *X, float *Y) {
  cusparseHandle_t handle = (cusparseHandle_t)wp::get_cusparse_handle();
  cusparseSetStream(handle, (cudaStream_t)cuda_stream_get_current());

  cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  float alpha = 1.f;

  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;

  cusparseSpSVDescr_t spsvDescr;

  cusparseCreateCsr(&matA, n, n, nnz, offsets, columns, values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  cusparseCreateDnVec(&vecX, n, X, CUDA_R_32F);
  cusparseCreateDnVec(&vecY, n, Y, CUDA_R_32F);

  cusparseSpSV_createDescr(&spsvDescr);

  cusparseFillMode_t fillmode_ = fillmode;
  cusparseDiagType_t diagtype_ = diagtype;

  cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE, &fillmode_,
                            sizeof(fillmode));
  cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype_,
                            sizeof(diagtype));

  size_t bufferSize = 0;
  void *buffer = nullptr;
  cusparseSpSV_bufferSize(handle, op, &alpha, matA, vecX, vecY, CUDA_R_32F,
                          CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, &bufferSize);

  cudaMalloc(&buffer, bufferSize); // preallocate?
  cusparseSpSV_analysis(handle, op, &alpha, matA, vecX, vecY, CUDA_R_32F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, buffer);
  cusparseSpSV_solve(handle, op, &alpha, matA, vecX, vecY, CUDA_R_32F,
                     CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr);

  cusparseDestroySpMat(matA);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseSpSV_destroyDescr(spsvDescr);
  cudaFree(buffer);
}

void csr_solve_lt_device(int n, int nnz, int *offsets, int *columns,
                         float *values, float *X, float *Y) {
  csr_solve_device_<CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT>(
      n, nnz, offsets, columns, values, X, Y);
}
