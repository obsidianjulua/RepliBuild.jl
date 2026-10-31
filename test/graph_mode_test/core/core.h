#ifndef CORE_H
#define CORE_H
#include "cbase.h"
#ifdef __cplusplus
extern "C" {
#endif

/* x * GM_VERSION + 1, through cbase_add */
int core_scale(int x);
/* cbase_bump, called from this library */
int core_bump(void);
int core_version(void);

#ifdef __cplusplus
}
#endif
#endif
