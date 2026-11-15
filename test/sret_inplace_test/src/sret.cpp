#include "sret.h"
#include <stdexcept>

SelfRef::SelfRef(int x) : self(this), v(x), w(x * 0.5) {}
SelfRef::SelfRef(const SelfRef &o) : self(this), v(o.v), w(o.w) {}
SelfRef::~SelfRef() { self = nullptr; }

SelfRef make_self(int x) { return SelfRef(x); }
int     self_ok(const SelfRef &s) { return s.self == &s ? 1 : 0; }
int     self_value(const SelfRef &s) { return s.v; }

Plain make_plain(int x) { return Plain{x, x + 1, x + 2}; }

std::string short_str(int x) { return std::string(2, (char)('a' + x)); }
int str_data_inside(const std::string &s) {
    const char *b = (const char *)&s, *d = s.data();
    return (d >= b && d < b + sizeof(s)) ? 1 : 0;
}
int str_size(const std::string &s) { return (int)s.size(); }
std::string  long_str(int n) { return std::string(n, 'x') + std::to_string(n); }
std::wstring wide_str(int n) { return std::wstring(n, L'w'); }

int str_size_v(std::string s) { return (int)s.size(); }
int kind_of(const std::string &s) { (void)s; return 1; }
int kind_of(const char *s) { (void)s; return 2; }
int str_throw(const std::string &s) {
    if (s == "boom") throw std::runtime_error("boom");
    return (int)s.size();
}
int str_both(const std::string &a, std::string b) { return (int)a.size() * 100 + (int)b.size(); }
