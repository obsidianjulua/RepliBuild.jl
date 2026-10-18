#ifndef ROUTING_H
#define ROUTING_H

/* Library-free fixture for how a C++ package routes a by-value record:
   Tier 3 (ccall) when the record's layout is natural and its Julia type is
   the C layout, Tier 2 (thunk) otherwise. The functions are extern "C", so
   exception routing does not apply and the record alone decides.

   Declared C++-style (no typedef), as the DWARF parser records a nested
   record member's size only through the record type itself. The
   size-as-alignment test that used to decide this sent CS3 and Mass to Tier 2,
   and let Top6 (packed two levels down) through to ccall. */

struct CS3 { char c; short s[3]; };                         /* natural, 8 B: Tier 3 */
struct Vec2 { float x, y; };
struct Mass { float m; Vec2 c; float i; };                  /* natural, 16 B of float: Tier 3 */
enum Kind : short { KIND_A = 1, KIND_B = 2 };
struct EnumRec { char c; Kind k; };                         /* natural, 4 B */
struct __attribute__((packed)) Pk5 { char c; int i; };      /* packed: Tier 2 */
struct __attribute__((packed)) In3 { char x; short y; };
struct Out6 { In3 in; char c[3]; };
struct Top6 { Out6 o; };                                    /* packed two levels down: Tier 2 */

extern "C" {

CS3     make_cs3(int base);
int     cs3_sum(CS3 v);
Mass    make_mass(float m);
float   mass_sum(Mass v);
EnumRec make_enumrec(int c);
int     enumrec_sum(EnumRec v);
Pk5     make_pk5(int i);
int     pk5_sum(Pk5 v);
Top6    make_top6(short y);
int     top6_sum(Top6 v);

}

#endif
