#include "core.h"
#include "gm_version.h"   /* written by a build-time step */

#ifndef GM_CORE_BUILD
#error "Core compiled without its own compile group"
#endif

int core_scale(int x) { return cbase_add(x * GM_VERSION, 1); }
int core_bump(void) { return cbase_bump(); }
int core_version(void) { return GM_VERSION; }
