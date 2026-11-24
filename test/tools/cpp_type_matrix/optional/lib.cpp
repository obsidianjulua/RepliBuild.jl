#include <optional>
#include <string>
std::optional<int> opt_make(int v, bool has) { return has ? std::optional<int>(v) : std::nullopt; }
int opt_or(std::optional<int> o, int d) { return o.value_or(d); }
int opt_or_ref(const std::optional<int>& o, int d) { return o.value_or(d); }
std::optional<double> optd_make(double v, bool has) { return has ? std::optional<double>(v) : std::nullopt; }
std::optional<std::string> opts_make(bool has) { return has ? std::optional<std::string>("text") : std::nullopt; }
int opts_len(const std::optional<std::string>& o) { return o ? (int)o->size() : -1; }
// 1 when a short string's characters are inside `obj`, where both libstdc++ and
// libc++ keep them. A value moved bytewise to another address fails this: its
// data pointer still names the place it was built in. Reads no character.
template <class S> static int short_string_inside(const S& s, const void* obj, size_t n) {
    if ((s.size() + 1) * sizeof(typename S::value_type) > 16) return 1;   // on the heap in both
    const char* d = (const char*)s.data(); const char* b = (const char*)obj;
    return (d >= b && d < b + n) ? 1 : 0;
}
int opts_intact(const std::optional<std::string>& o) { return o ? short_string_inside(*o, &o, sizeof(o)) : 1; }
// The same check on a value that never left C++: what the probe compares with.
int opts_intact_self() { auto o = opts_make(true); return opts_intact(o); }
