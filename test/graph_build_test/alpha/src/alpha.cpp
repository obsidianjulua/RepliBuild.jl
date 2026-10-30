// alpha: a C++ shared library (a class and the STL inside) with a C entry point.
#include "alpha.h"

#include <numeric>
#include <vector>

// Each translation unit compiles under its own target's compile group.
#ifndef ALPHA_FACTOR
#error "alpha's own define is missing"
#endif
#ifdef BETA_OFFSET
#error "beta's define reached alpha"
#endif

namespace {
class Scaler {
public:
    explicit Scaler(int factor) : factor_(factor) {}
    int apply(int x) const {
        std::vector<int> v(static_cast<std::size_t>(factor_), x);
        return std::accumulate(v.begin(), v.end(), 0);
    }

private:
    int factor_;
};
}  // namespace

extern "C" int alpha_scale(int x) { return Scaler(ALPHA_FACTOR).apply(x); }
