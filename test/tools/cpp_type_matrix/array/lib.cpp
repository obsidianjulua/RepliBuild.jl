#include <array>
std::array<int, 4> arr_make(int k) { return {k, k + 1, k + 2, k + 3}; }
int arr_sum(std::array<int, 4> a) { return a[0] + a[1] + a[2] + a[3]; }
int arr_sum_ref(const std::array<int, 4>& a) { return a[0] + a[1] + a[2] + a[3]; }
std::array<double, 3> arrd_make(double k) { return {k, 2 * k, 3 * k}; }
double arrd_sum(std::array<double, 3> a) { return a[0] + a[1] + a[2]; }
void arr_fill(std::array<int, 4>& a, int v) { a.fill(v); }
