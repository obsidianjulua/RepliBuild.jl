#include <complex>
std::complex<double> cx_make(double re, double im) { return {re, im}; }
double cx_abs(std::complex<double> z) { return std::abs(z); }
double cx_abs_ref(const std::complex<double>& z) { return std::abs(z); }
std::complex<float> cxf_make(float re, float im) { return {re, im}; }
float cxf_re(std::complex<float> z) { return z.real(); }
