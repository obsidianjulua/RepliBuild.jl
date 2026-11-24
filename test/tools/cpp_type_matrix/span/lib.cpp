#include <span>
#include <vector>
int span_sum(std::span<const int> s) { int t = 0; for (int x : s) t += x; return t; }
int span_sum_ref(const std::span<const int>& s) { int t = 0; for (int x : s) t += x; return t; }
void span_fill(std::span<int> s, int v) { for (int& x : s) x = v; }
static int g_buf[4] = {1, 2, 3, 4};
std::span<int> span_global() { return std::span<int>(g_buf, 4); }
int span_fixed(std::span<const int, 3> s) { return s[0] + s[1] + s[2]; }
