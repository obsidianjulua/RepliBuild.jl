#include <filesystem>
#include <string>
std::filesystem::path path_join(const std::filesystem::path& a, const std::string& b) { return a / b; }
int path_len(const std::filesystem::path& p) { return (int)p.string().size(); }
std::string path_ext(std::filesystem::path p) { return p.extension().string(); }
std::filesystem::path path_make(const char* s) { return std::filesystem::path(s); }
// 1 when a short string's characters are inside `obj`, where both libstdc++ and
// libc++ keep them. A value moved bytewise to another address fails this: its
// data pointer still names the place it was built in. Reads no character.
template <class S> static int short_string_inside(const S& s, const void* obj, size_t n) {
    if ((s.size() + 1) * sizeof(typename S::value_type) > 16) return 1;   // on the heap in both
    const char* d = (const char*)s.data(); const char* b = (const char*)obj;
    return (d >= b && d < b + n) ? 1 : 0;
}
int path_intact(const std::filesystem::path& p) { return short_string_inside(p.native(), &p, sizeof(p)); }
// The same check on a value that never left C++: what the probe compares with.
int path_intact_self() { auto p = path_make("a/b.txt"); return path_intact(p); }
