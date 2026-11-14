// A pass-by-reference record returned by value through a Tier 2 thunk
// (test_sret_in_caller_buffer.jl, 2026-11-13).
//
// SelfRef stores its own address in `self`, set by every constructor. Its
// user-provided copy constructor and destructor make it non-trivial for the
// purposes of calls (DW_CC_pass_by_reference), so C++ returns it through a
// hidden pointer and builds it there. The thunk used to pass its own alloca
// as that pointer and copy the bytes out, so `self` came back aiming at the
// thunk's dead frame. Built in the caller's buffer, `self` is that buffer.
#pragma once
#include <string>

struct SelfRef {
    SelfRef *self;
    int v;
    double w;
    SelfRef(int x);
    SelfRef(const SelfRef &o);
    ~SelfRef();
};

SelfRef make_self(int x);                 // C++ linkage, not noexcept: Tier 2
int     self_ok(const SelfRef &s);         // 1 when s.self == &s, else 0
int     self_value(const SelfRef &s);

// An STL container returned by value is pass-by-reference too, on its own
// `:direct` path. A short string keeps its characters inside the object, so
// built in place, its data pointer points into the caller's buffer.
std::string short_str(int x);
int         str_data_inside(const std::string &s);   // 1 when s.data() is inside s
int         str_size(const std::string &s);

// Control: trivially copyable, returned by value the way it always was.
struct Plain { int a, b, c; };
Plain make_plain(int x);
