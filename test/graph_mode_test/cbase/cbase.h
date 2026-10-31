#ifndef CBASE_H
#define CBASE_H
#ifdef __cplusplus
extern "C" {
#endif

int cbase_add(int a, int b);
/* One counter per loaded copy of the library: every library of the graph
   bumps the same one only when the process holds one copy of cbase. */
int cbase_bump(void);
int cbase_count(void);

#ifdef __cplusplus
}
#endif
#endif
