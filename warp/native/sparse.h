#pragma once

#include "builtin.h"
#include <cusparse_v2.h>

namespace wp {

bool init_cusparse();
void destroy_cusparse();
cusparseHandle_t get_cusparse_handle();

} // namespace wp
