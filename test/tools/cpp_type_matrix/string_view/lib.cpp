#include <string_view>
int sv_len(std::string_view s) { return (int)s.size(); }
int sv_len_ref(const std::string_view& s) { return (int)s.size(); }
std::string_view sv_tail(std::string_view s, int n) { return s.substr(s.size() - (size_t)n); }
std::string_view sv_literal() { return "literal"; }
char sv_at(std::string_view s, int i) { return s[(size_t)i]; }
