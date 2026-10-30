#include <core/core.h>
#include "ext_version.h"
extern "C" int helper_twice(int);
int ext_value() { return helper_twice(core_value()) + EXT_LEVEL + EXT_VERSION; }
