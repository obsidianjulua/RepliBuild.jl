/* ca: a C shared library with state of its own. */
#include "ca.h"

static int counter;

int ca_next(int x) { return x + (++counter); }
