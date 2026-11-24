#include <string>
std::string str_make(int n) { return std::string(n, 'a'); }
int str_len_ref(const std::string& s) { return (int)s.size(); }
int str_len_val(std::string s) { return (int)s.size(); }
void str_append(std::string& s, const char* t) { s += t; }
static std::string g_str = "global text";
const std::string& str_global() { return g_str; }
std::string* str_global_ptr() { return &g_str; }
