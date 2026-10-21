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

/* Byte regions by value through a thunk (abi_fuzz X2). A union and a bitfield
   record have no field-by-field body, so the thunk passes their bytes, which
   must still reach the callee the way C passes the record, and come back the
   way C returns it into the record Julia allocates (X1). */
union U4 { int i; unsigned char b[4]; };                    /* 4 B: one register */
union U12 { int i[3]; char c; };                            /* 12 B: two eightbytes (SysV), by reference (Win64) */
struct BF8 { unsigned a : 3; unsigned b : 7; int n; };      /* 8 B bitfield record */

/* A `#pragma pack` record holding a pointer-aligned one (abi_fuzz X8): `r` sits
   at offset 4, where natural alignment would put it at 8. */
struct PIn { int a; void* p; };
struct PR { int m1; PIn in; };
#pragma pack(push, 4)
struct PK28 { char c; PR r; };
#pragma pack(pop)

/* A tiled record holding a bitfield (abi_fuzz X9): the member sizes sum to the
   record's, so the thunk body tiles them, and the bitfield's unit must be read
   where that body puts it. 12 bytes on MinGW (MS-style: m2 has its own 8-byte
   unit at 4), 5 on Itanium; m2's bits are in byte 4 either way. */
#pragma pack(push, 1)
struct PB12 { unsigned int m1; unsigned long long m2 : 7; };
#pragma pack(pop)

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
int     u4_sum(U4 v);
int     u12_sum(U12 v);
int     bf8_sum(BF8 v);
U4      make_u4(int i);
U12     make_u12(int base);
BF8     make_bf8(int n);
PK28    make_pk28(int m1);
int     pk28_sum(PK28 v);
int     pb12_sum(PB12 v);
PB12    make_pb12(int m1);

}

#endif
