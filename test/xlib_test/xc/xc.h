#ifndef XC_H
#define XC_H
#ifdef __cplusplus
extern "C" {
#endif

/* A POD two libraries pass by value: 16 bytes, INTEGER then SSE on SysV. */
typedef struct { int a; double b; } Pt;

enum Mode { MODE_A = 1, MODE_B = 2, MODE_C = 3 };

Pt xc_make(int a, double b);
double xc_sum(Pt p);
int xc_mode(enum Mode m);

#ifdef __cplusplus
}
#endif
#endif
