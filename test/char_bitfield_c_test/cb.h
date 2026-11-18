/* C side of test_char_bitfield_live.jl (fixture char_bitfield_c_test): `char` by value, signed bitfields,
   an open enum taken by value. */
#ifndef CB_H
#define CB_H

typedef struct {
    unsigned a : 3;
    int s : 7;           /* signed, one byte */
    unsigned u : 5;
    long long w : 40;    /* signed, spans six bytes */
    signed char t : 4;   /* signed, declared signed char */
} Bits;

enum Mode { MODE_LOW = -1, MODE_MID = 0, MODE_HIGH = 1 };
typedef struct { enum Mode m : 3; unsigned k : 2; } ModeBits;   /* enum of int */

typedef struct { char tag; } OneByte;

int     cb_char_value(char c);          /* (int)c: shows the caller's extension */
char    cb_char_next(char c);
int     cb_schar_value(signed char c);
OneByte cb_onebyte(char c);
void    cb_bits_fill(Bits *out);        /* a=5, s=-3, u=17, w=-123456789012, t=-8 */
int     cb_bits_s(Bits b);
long long cb_bits_w(Bits b);
int     cb_bits_t(Bits b);
void    cb_modebits_fill(ModeBits *out);  /* m=MODE_LOW, k=2 */
int     cb_mode_value(enum Mode m);
#endif
