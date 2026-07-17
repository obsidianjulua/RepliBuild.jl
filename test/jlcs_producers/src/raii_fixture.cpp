// Fixture for the JLCS producer tests (scope-RAII + array-view).
//
// Grip is deliberately non-trivial for the purposes of calls (user-provided
// copy ctor + dtor): under the Itanium ABI, by-value Grip arguments are passed
// by reference to a caller-owned temporary. The extern-C tally makes ctor/dtor
// firing observable from Julia.
//
// Built with -O0 so the C1/C2 and D1/D2 symbols are all emitted un-inlined.

#include <cstdint>

extern "C" {
int64_t jlcs_tally = 0;
int64_t jlcs_get_tally() { return jlcs_tally; }
void jlcs_reset_tally() { jlcs_tally = 0; }
}

// Out-of-line definitions: inline ctors/dtors are only emitted when odr-used,
// and under Itanium the CALLER destroys by-value temporaries — so consume_grip
// alone references neither. Out-of-line makes them strong T symbols.
struct Grip {
    int64_t v;
    Grip(int64_t x);
    Grip(const Grip& o);
    ~Grip();
};
Grip::Grip(int64_t x) : v(x) {}
Grip::Grip(const Grip& o) : v(o.v) { jlcs_tally += 100; }
Grip::~Grip() { jlcs_tally += 1; }

// By-value non-trivial parameter: callee receives Grip* per Itanium ABI.
// Mutates its (copy of the) argument to prove the caller's object is isolated.
int64_t consume_grip(Grip g) {
    g.v += 1000;   // must never be visible in the caller's object
    return g.v * 2;
}
