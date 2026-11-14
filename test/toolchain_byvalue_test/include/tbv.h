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

// Laid out from their bases (2026-11-13). libstdc++'s tuple and optional have
// no member of their own, only bases, so they had no layout and trapped:
// tuple<int, double> holds the double at 0 and the int at 8, through two
// `_Head_base`s; optional<T> holds a `_Storage` union (T beside an empty
// `_Empty_byte`) and an engaged flag. complex's member is a `__complex__`
// scalar, whose width the parser reads off its byte_size.
std::tuple<int, double> tup(int a, double b);
std::optional<double>   maybe(bool has, double v);
std::optional<int>      maybe_int(bool has, int v);
std::complex<double>    cplx(double re, double im);
std::complex<float>     cplxf(float re, float im);
double tup_sum(std::tuple<int, double> t);              // the same records as parameters
double maybe_or(std::optional<double> o, double fallback);
int    maybe_int_or(std::optional<int> o, int fallback);
double cplx_norm(std::complex<double> z);
float  cplxf_norm(std::complex<float> z);

// A bare `long double _Complex` comes back in st0/st1, which nothing can call;
// this record holding one is 32 bytes whose first eightbyte is X87, so SysV
// returns it in memory (sret), and it is callable.
std::complex<long double> cplxl(double re, double im);

// Controls with no toolchain type in them.
struct Tag {};
struct LP : Tag { double a, b; };                      // an empty base, like std::pair's __pair_base
LP     make_lp(double a, double b);
double lp_sum(LP p);

template <class T> struct Box { T a, b; };
struct Outer3 { double x; Box<double> in; };           // a template record inside a MEMORY return
Outer3 make_outer3(double x, double a, double b);

// An exported global holding a toolchain record by value. The test's TOML drops
// it with `[wrap] exclude_symbols`, so wrap will not bind it and it seeds no
// toolchain record: `pair<long, long>` stays out (glfw's `_glfw`, 2026-11-08).
struct Ledger { std::pair<long, long> span; int n; };
extern Ledger tbv_ledger;
