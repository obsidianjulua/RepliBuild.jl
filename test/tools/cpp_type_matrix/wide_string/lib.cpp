#include <string>
std::wstring ws_make(int n) { return std::wstring((size_t)n, L'w'); }
int ws_len(const std::wstring& s) { return (int)s.size(); }
std::u16string u16_make(int n) { return std::u16string((size_t)n, u'x'); }
int u16_len(const std::u16string& s) { return (int)s.size(); }
std::u8string u8_make(int n) { return std::u8string((size_t)n, u8'x'); }
// 1 when a short string's characters are inside `obj`, where both libstdc++ and
// libc++ keep them. A value moved bytewise to another address fails this: its
// data pointer still names the place it was built in. Reads no character.
template <class S> static int short_string_inside(const S& s, const void* obj, size_t n) {
    if ((s.size() + 1) * sizeof(typename S::value_type) > 16) return 1;   // on the heap in both
    const char* d = (const char*)s.data(); const char* b = (const char*)obj;
    return (d >= b && d < b + n) ? 1 : 0;
}
int ws_intact(const std::wstring& s) { return short_string_inside(s, &s, sizeof(s)); }
int u16_intact(const std::u16string& s) { return short_string_inside(s, &s, sizeof(s)); }
// The same check on a value that never left C++: what the probe compares with.
int ws_intact_self() { auto s = ws_make(3); return ws_intact(s); }
int u16_intact_self() { auto s = u16_make(4); return u16_intact(s); }
