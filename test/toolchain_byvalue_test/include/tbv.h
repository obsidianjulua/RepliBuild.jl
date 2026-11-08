// Toolchain records crossing by value (test_toolchain_byvalue_live.jl).
//
// Every type below except Span, Tag, LP, Box and Outer3 is declared in a
// libstdc++/libc++ header, so parse_dwarf_dump's provenance gate keeps it out of
// struct_definitions, and before 2026-10-29 every function here was an ABI trap.
#pragma once
#include <array>
#include <complex>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <map>

// std::pair in each SysV shape
std::pair<double, double> minmax_pair(double a, double b);             // SSE, SSE
std::pair<int, double>    tagged(int k, double v);                     // INTEGER, SSE
std::pair<int, int>       divmod(int a, int b);                        // one INTEGER
std::pair<float, float>   fpair(float a, float b);                     // one SSE, two floats
std::pair<double, std::pair<double, double>> triple(double a, double b, double c);  // MEMORY (sret)
std::pair<std::pair<int, int>, double> nested(int a, int b, double c); // a pair inside a pair
std::pair<int *, bool>    find_first(int *xs, int n, int want);        // a `*` inside the arguments
double pair_sum(std::pair<double, double> p);                          // by-value parameters
int    pair_imix(std::pair<int, double> p, int scale);

// A library record holding a toolchain record by value.
struct Span { std::pair<int, int> range; double weight; };
Span   make_span(int lo, int hi, double w);
double span_weight(Span s);

// Other trivially copyable toolchain records with a layout.
std::array<double, 2> arr2(double a, double b);
std::string_view      tail_view(const char *s, int n);

// Not the wrapper's to reach, so their records stay out (the dynamic-symbol
// gate): a hidden function's by-value pair<short, short>, and the libstdc++
// map instantiations map_lookup pulls in, which return _Rb_tree_iterator by
// value and are in dynsym because `namespace std` has default visibility.
__attribute__((visibility("hidden"))) std::pair<short, short> hidden_pair(short a);
int map_lookup(int k);

// Still traps: no layout to keep (tuple and optional are bases all the way
// down in libstdc++; complex's member is `__complex__ double`).
std::tuple<int, double> tup(int a, double b);
std::optional<double>   maybe(bool has, double v);
std::complex<double>    cplx(double re, double im);

// Controls with no toolchain type in them.
struct Tag {};
struct LP : Tag { double a, b; };                      // an empty base, like std::pair's __pair_base
LP     make_lp(double a, double b);
double lp_sum(LP p);

template <class T> struct Box { T a, b; };
struct Outer3 { double x; Box<double> in; };           // a template record inside a MEMORY return
Outer3 make_outer3(double x, double a, double b);
