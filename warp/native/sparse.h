#pragma once

#include "builtin.h"

namespace wp {

bool init_cusparse();
void destroy_cusparse();
void* get_cusparse_handle();

} // namespace wp
