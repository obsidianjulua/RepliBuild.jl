#include <functional>
int fn_apply(std::function<int(int)> f, int x) { return f(x); }
int fn_apply_ref(const std::function<int(int)>& f, int x) { return f(x); }
std::function<int(int)> fn_adder(int k) { return [k](int x) { return x + k; }; }
int fn_empty(const std::function<int(int)>& f) { return f ? 0 : 1; }
int fn_each(const std::function<void(int)>& f, int n) { for (int i = 0; i < n; i++) f(i); return n; }
double fn_fold(const std::function<double(double, double)>& f, double a, double b) { return f(a, b); }
typedef int (*plain_cb)(int);
int fn_plain(plain_cb f, int x) { return f(x); }
