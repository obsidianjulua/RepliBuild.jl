#include "unwind_api.h"

/* A C source in the C++ library: compiled as C, so its definition is
   `nounwind` and the function keeps its plain ccall. */
int cl_plain_c(int x) { return x * 3; }
