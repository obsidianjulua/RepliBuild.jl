#ifndef UNWIND_API_H
#define UNWIND_API_H

/* Library-free fixture for C linkage in a C++ library (2026-11-17).
   `extern "C"` gives a function a C name; its body is still C++ and may
   throw. A throw through a plain `ccall` has no landing pad and ends the
   process in std::terminate, so such a function may take `ccall` only when
   its definition is `nounwind` in the linked IR. The `cl_*` functions in
   unwind_api.cpp throw, or call something the compiler cannot see through;
   `cl_add` cannot throw, and `cl_plain_c` (unwind_plain.c) is compiled as C. */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ClPoint { double x; double y; } ClPoint;

int         cl_checked(int x);                   /* x < 0: throws std::runtime_error */
void        cl_checked_void(int x);              /* x < 0: throws std::out_of_range */
const char *cl_name(int which);                  /* which < 0: throws std::invalid_argument */
ClPoint     cl_point(double x, double y);        /* x < 0: throws std::domain_error */
int         cl_apply(int (*op)(int), int x);     /* calls `op`; x < 0: throws std::logic_error */
int         cl_throws_int(int x);                /* x < 0: throws 42 */
int         cl_add(int a, int b);                /* cannot throw */
int         cl_calls(void);                      /* how many cl_* calls ran to their end */
int         cl_plain_c(int x);                   /* C source: x * 3 */

#ifdef __cplusplus
}
#endif

#endif
