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
// Read into a Julia String and ended there (StdStrings.take_string!): a heap
// string's buffer is freed by its own destructor. A wide string keeps its raw
// bytes, with the reason in the wrapper.
std::string  long_str(int n);
std::wstring wide_str(int n);

// std::string PARAMETERS (2026-11-15): the wrapper builds a real std::string
// from a Julia string in a slot of its own, passes its address (by `const&`
// as the pointer; by value, DW_CC_pass_by_reference makes it the caller's
// temporary) and ends it after the call, whether the call returns or throws.
int str_size_v(std::string s);                        // by value
int kind_of(const std::string &s);                    // 1: the overload pair a Julia String
int kind_of(const char *s);                           // 2: and a Cstring tell apart
int str_throw(const std::string &s);                  // throws when s == "boom", else s.size()
int str_both(const std::string &a, std::string b);    // a.size() * 100 + b.size()

// Control: trivially copyable, returned by value the way it always was.
struct Plain { int a, b, c; };
Plain make_plain(int x);
