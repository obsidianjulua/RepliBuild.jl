#include "cbase.h"

#ifdef GM_CORE_BUILD
#error "cbase compiled under Core's compile group"
#endif

static int counter = 0;

int cbase_add(int a, int b) { return a + b; }
int cbase_bump(void) { return ++counter; }
int cbase_count(void) { return counter; }
