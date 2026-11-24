#include <vector>
#include <string>
#include "types.hpp"
std::vector<int> vec_make(int n) { std::vector<int> v; for (int i = 1; i <= n; i++) v.push_back(i); return v; }
int vec_sum_ref(const std::vector<int>& v) { int s = 0; for (int x : v) s += x; return s; }
int vec_sum_val(std::vector<int> v) { int s = 0; for (int x : v) s += x; return s; }
void vec_push(std::vector<int>& v, int x) { v.push_back(x); }
std::vector<double> vecd_make(int n) { return std::vector<double>((size_t)n, 0.5); }
double vecd_sum(const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return s; }
std::vector<P2> vecp_make(int n) { std::vector<P2> v; for (int i = 0; i < n; i++) v.push_back(P2{i, 2 * i}); return v; }
int vecp_sum(const std::vector<P2>& v) { int s = 0; for (auto& p : v) s += p.x + p.y; return s; }
std::vector<std::string> vecs_make(int n) { return std::vector<std::string>((size_t)n, "ab"); }
int vecs_total(const std::vector<std::string>& v) { int s = 0; for (auto& x : v) s += (int)x.size(); return s; }
std::vector<std::vector<int>> vecv_make(int n) { return std::vector<std::vector<int>>((size_t)n, std::vector<int>{1, 2}); }
int vecv_total(const std::vector<std::vector<int>>& v) { int s = 0; for (auto& r : v) for (int x : r) s += x; return s; }
std::vector<bool> vecb_make(int n) { return std::vector<bool>((size_t)n, true); }
int vecb_count(const std::vector<bool>& v) { int s = 0; for (bool b : v) s += b; return s; }
