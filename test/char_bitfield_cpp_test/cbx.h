// C++ side of test_char_bitfield_live.jl: `char` by value through a thunk and
// through ccall, `char` in an overload set, a signed bitfield record.
#pragma once

namespace cbx {
struct Flags {
    unsigned on : 1;
    int level : 4;       // signed
    unsigned rest : 11;
};

int  char_value(char c);                 // not noexcept: Tier 2
char char_next(char c);                  // Tier 2
int  char_value_nx(char c) noexcept;     // noexcept: ccall

int ovl(int x);                          // tags: 1, 2, 3
int ovl(long x);
int ovl(char c);

void flags_fill(Flags* out);             // on=1, level=-5, rest=1000
int  flags_level(const Flags& f);
}
