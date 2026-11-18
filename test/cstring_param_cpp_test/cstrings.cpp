// test_cstring_params_live.jl: pointers to `char` through a C++ wrapper. A
// `noexcept` function and an `extern "C"` function proven `nounwind` are
// plain ccalls; one that may throw takes a Tier 2 thunk. All three take a
// Julia string with `Cstring` semantics, and only a pointer to plain `char`
// is a C string, in a parameter and in a return.
#include <cstddef>
#include <stdexcept>

static std::size_t count(const char *s) {
    std::size_t n = 0;
    while (s[n]) n++;
    return n;
}

std::size_t cpp_len(const char *s) noexcept { return count(s); }

// No call in the body, so the definition is `nounwind` at any level.
extern "C" std::size_t c_len(const char *s) {
    std::size_t n = 0;
    while (s[n]) n++;
    return n;
}

std::size_t thunk_len(const char *s) {
    if (!s) throw std::invalid_argument("thunk_len: null");
    return count(s);
}

static const signed char g_sbytes[] = {-1, -2, 0};
const signed char *cpp_sbytes() noexcept { return g_sbytes; }
const signed char *thunk_sbytes(int which) {
    if (which < 0) throw std::out_of_range("thunk_sbytes: negative");
    return g_sbytes;
}
const char *cpp_greet() noexcept { return "hello"; }
