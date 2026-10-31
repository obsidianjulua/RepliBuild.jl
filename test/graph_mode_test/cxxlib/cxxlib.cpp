#include "cxxlib.h"
#include "core.h"

#ifdef GM_CORE_BUILD
#error "cxxlib compiled under Core's compile group"
#endif

// The live test edits this value and checks that cxxlib alone relinks.
#define GM_CXX_OFFSET 100

namespace gm {
int total(int x) noexcept { return core_scale(x) + cbase_add(x, GM_CXX_OFFSET); }
}

extern "C" int cxx_total(int x) noexcept { return 2 * gm::total(x); }
