#include <any>
std::any any_int(int v) { return v; }
int any_get(const std::any& a) { return a.type() == typeid(int) ? std::any_cast<int>(a) : -1; }
int any_has(std::any a) { return a.has_value() ? 1 : 0; }
