#include "tbv.h"
#include <cstring>

std::pair<double, double> minmax_pair(double a, double b) { return a < b ? std::make_pair(a, b) : std::make_pair(b, a); }
std::pair<int, double>    tagged(int k, double v) { return {k, v * 2}; }
std::pair<int, int>       divmod(int a, int b) { return {a / b, a % b}; }
std::pair<float, float>   fpair(float a, float b) { return {a + b, a - b}; }
std::pair<double, std::pair<double, double>> triple(double a, double b, double c) { return {a, {b, c}}; }
std::pair<std::pair<int, int>, double> nested(int a, int b, double c) { return {{a, b}, c}; }
std::pair<int *, bool> find_first(int *xs, int n, int want) {
    for (int i = 0; i < n; ++i)
        if (xs[i] == want) return {xs + i, true};
    return {nullptr, false};
}
double pair_sum(std::pair<double, double> p) { return p.first + p.second; }
int    pair_imix(std::pair<int, double> p, int scale) { return p.first * scale + (int)p.second; }

Span   make_span(int lo, int hi, double w) { return Span{{lo, hi}, w}; }
double span_weight(Span s) { return (s.range.second - s.range.first) * s.weight; }

std::array<double, 2> arr2(double a, double b) { return {a, b}; }
std::string_view      tail_view(const char *s, int n) {
    size_t l = std::strlen(s);
    return std::string_view(s + l - n, n);
}

std::tuple<int, double> tup(int a, double b) { return {a, b}; }
std::optional<double>   maybe(bool has, double v) { if (has) return v; return std::nullopt; }
std::complex<double>    cplx(double re, double im) { return {re, im}; }

LP     make_lp(double a, double b) { LP p; p.a = a; p.b = b; return p; }
double lp_sum(LP p) { return p.a + p.b; }

Outer3 make_outer3(double x, double a, double b) { return Outer3{x, {a, b}}; }

std::pair<short, short> hidden_pair(short a) { return {a, (short)(a + 1)}; }
int map_lookup(int k) {
    static std::map<int, int> m{{1, 10}, {2, 20}};
    auto it = m.find(k);
    return it == m.end() ? -1 : it->second + hidden_pair((short)k).second;
}
