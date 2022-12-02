#pragma once

#include "builtin.h"

namespace wp {

bool init_cublas();
void destroy_cublas();
void* get_cublas_handle();

bool init_cusparse();
void destroy_cusparse();
void* get_cusparse_handle();

} // namespace wp
