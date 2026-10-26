/* A library source that includes the header without the define: it links
   against the implementation the shim file compiles, with no copy of its own. */
#include "stbish.h"

int user_total(void) { return stbish_add(40, 2); }
