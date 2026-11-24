#include <stdexcept>
#include <string>
#include <system_error>
int ex_runtime(int x) { if (x < 0) throw std::runtime_error("negative input"); return x; }
int ex_out_of_range(int x) { if (x > 9) throw std::out_of_range("too large"); return x; }
int ex_custom(int x);
struct MyError : std::exception { const char* what() const noexcept override { return "my error"; } };
int ex_custom(int x) { if (x == 0) throw MyError(); return x; }
int ex_int(int x) { if (x == 0) throw 42; return x; }
std::error_code ec_make(int v) { return std::error_code(v, std::generic_category()); }
int ec_value(std::error_code e) { return e.value(); }
