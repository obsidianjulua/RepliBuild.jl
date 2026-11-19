#include "cb.h"

int  cb_char_value(char c) { return c; }
char cb_char_next(char c) { return (char)(c + 1); }
int  cb_schar_value(signed char c) { return c; }
OneByte cb_onebyte(char c) { OneByte o = { c }; return o; }
void cb_bits_fill(Bits *out) {
    out->a = 5; out->s = -3; out->u = 17; out->w = -123456789012LL; out->t = -8;
}
int cb_bits_s(Bits b) { return b.s; }
long long cb_bits_w(Bits b) { return b.w; }
int cb_bits_t(Bits b) { return b.t; }
void cb_modebits_fill(ModeBits *out) { out->m = MODE_LOW; out->k = 2; }
int cb_mode_value(enum Mode m) { return (int)m * 10; }
