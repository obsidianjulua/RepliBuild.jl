#include <stdio.h>

int main() {
  printf("Hello, World!\n");
  return 0;
}

/* scalar integer add — pure call overhead */
int iadd(int a, int b) {
  return a + b;
}

/* fused multiply-add — a bit of work per call */
double fmadd(double a, double b, double c) {
  return a * b + c;
}

/* tighter loop kernel: horizontal sum of n doubles */
double hsum(const double* v, int n) {
  double s = 0.0;
  for (int i = 0; i < n; i++) s += v[i];
  return s;
}