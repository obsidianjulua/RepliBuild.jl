#include <utility>
#include <tuple>
#include <string>
std::pair<int, double> pair_make(int a, double b) { return {a, b}; }
double pair_sum(std::pair<int, double> p) { return p.first + p.second; }
double pair_sum_ref(const std::pair<int, double>& p) { return p.first + p.second; }
std::pair<std::string, int> pairs_make(int n) { return {std::string("key"), n}; }
int pairs_len(const std::pair<std::string, int>& p) { return (int)p.first.size() + p.second; }
std::tuple<int, double, char> tup_make(int a, double b, char c) { return {a, b, c}; }
double tup_sum(std::tuple<int, double, char> t) { return std::get<0>(t) + std::get<1>(t) + std::get<2>(t); }
double tup_sum_ref(const std::tuple<int, double, char>& t) { return std::get<0>(t) + std::get<1>(t) + std::get<2>(t); }
// 1 when a short string's characters are inside `obj`, where both libstdc++ and
// libc++ keep them. A value moved bytewise to another address fails this: its
// data pointer still names the place it was built in. Reads no character.
template <class S> static int short_string_inside(const S& s, const void* obj, size_t n) {
    if ((s.size() + 1) * sizeof(typename S::value_type) > 16) return 1;   // on the heap in both
    const char* d = (const char*)s.data(); const char* b = (const char*)obj;
    return (d >= b && d < b + n) ? 1 : 0;
}
int pairs_intact(const std::pair<std::string, int>& p) { return short_string_inside(p.first, &p, sizeof(p)); }
// The same check on a value that never left C++: what the probe compares with.
int pairs_intact_self() { auto p = pairs_make(4); return pairs_intact(p); }
