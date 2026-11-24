#include <variant>
#include <string>
std::variant<int, double> var_make(int which) { if (which == 0) return 7; return 2.5; }
int var_index(std::variant<int, double> v) { return (int)v.index(); }
double var_value(const std::variant<int, double>& v) { return v.index() == 0 ? (double)std::get<int>(v) : std::get<double>(v); }
std::variant<int, std::string> vars_make(int which) { if (which == 0) return 7; return std::string("text"); }
int vars_index(const std::variant<int, std::string>& v) { return (int)v.index(); }
