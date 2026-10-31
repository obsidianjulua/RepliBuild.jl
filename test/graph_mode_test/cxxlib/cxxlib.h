#ifndef CXXLIB_H
#define CXXLIB_H

namespace gm {
// core_scale(x) + cbase_add(x, GM_CXX_OFFSET): one call into each library below.
int total(int x) noexcept;
}

extern "C" int cxx_total(int x) noexcept;   // 2 * gm::total(x)

#endif
