#include "cbx.h"

namespace cbx {
int  char_value(char c) { return c; }
char char_next(char c) { return (char)(c + 1); }
int  char_value_nx(char c) noexcept { return c; }

int ovl(int x) { return 1000 + x; }
int ovl(long x) { return 2000 + (int)x; }
int ovl(char c) { return 3000 + c; }

void flags_fill(Flags* out) { out->on = 1; out->level = -5; out->rest = 1000; }
int  flags_level(const Flags& f) { return f.level; }
}
