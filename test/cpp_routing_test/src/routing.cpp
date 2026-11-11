#include "routing.h"

/* Each *_sum weights every member differently, so a member read from the wrong
   place changes the answer. */

extern "C" {

CS3 make_cs3(int base)
{
    CS3 r = { (char)(base + 1), { (short)(base * 2), (short)(base * 3), (short)(-base) } };
    return r;
}

int cs3_sum(CS3 v)
{
    return v.c + v.s[0] * 10 + v.s[1] * 100 + v.s[2] * 1000;
}

Mass make_mass(float m)
{
    Mass r = { m, { m + 1.0f, m + 2.0f }, m * 4.0f };
    return r;
}

float mass_sum(Mass v)
{
    return v.m + 10.0f * v.c.x + 100.0f * v.c.y + 1000.0f * v.i;
}

EnumRec make_enumrec(int c)
{
    EnumRec r = { (char)c, KIND_B };
    return r;
}

int enumrec_sum(EnumRec v)
{
    return v.c + 100 * (int)v.k;
}

Pk5 make_pk5(int i)
{
    Pk5 r = { 0x21, i };
    return r;
}

int pk5_sum(Pk5 v)
{
    return v.c + 2 * v.i;
}

Top6 make_top6(short y)
{
    Top6 r = { { { 0x31, y }, { 4, 5, 6 } } };
    return r;
}

int top6_sum(Top6 v)
{
    return v.o.in.x + 10 * v.o.in.y + 100 * (v.o.c[0] + v.o.c[1] + v.o.c[2]);
}

int u4_sum(U4 v)
{
    return v.b[0] + 2 * v.b[1] + 3 * v.b[2] + 4 * v.b[3];
}

int u12_sum(U12 v)
{
    return v.i[0] + 2 * v.i[1] + 3 * v.i[2];
}

int bf8_sum(BF8 v)
{
    return (int)v.a + 10 * (int)v.b + 100 * v.n;
}

U4 make_u4(int i)
{
    U4 u;
    u.i = i;
    return u;
}

U12 make_u12(int base)
{
    U12 u;
    u.i[0] = base;
    u.i[1] = base + 1;
    u.i[2] = base + 2;
    return u;
}

BF8 make_bf8(int n)
{
    BF8 v = {};
    v.a = 5;
    v.b = 9;
    v.n = n;
    return v;
}

PK28 make_pk28(int m1)
{
    PK28 r = {};
    r.c = 0x41;
    r.r.m1 = m1;
    r.r.in.a = 7;
    r.r.in.p = (void*)0x1234;
    return r;
}

int pk28_sum(PK28 v)
{
    return v.c + 2 * v.r.m1 + 3 * v.r.in.a + (int)(long long)v.r.in.p;
}

int pb12_sum(PB12 v)
{
    return (int)v.m1 + 1000 * (int)v.m2;
}

PB12 make_pb12(int m1)
{
    PB12 r = {};
    r.m1 = (unsigned)m1;
    r.m2 = 11;
    return r;
}

MA32 make_ma32(double x)
{
    MA32 r = { x, x + 1.0, x + 2.0, x * 4.0 };
    return r;
}

/* The pad is weighed too, so a record read from the pad's slot is caught. */
double ma32_sum(Big24 pad, MA32 v)
{
    return (double)(pad.x + 2 * pad.y + 3 * pad.z) +
           v.a + 10.0 * v.b + 100.0 * v.c + 1000.0 * v.d;
}

NestMA make_nestma(double x)
{
    NestMA r = { make_ma32(x) };
    return r;
}

double nestma_sum(Big24 pad, NestMA v)
{
    return ma32_sum(pad, v.m);
}

/* v's high half is k + 1, so a 64-bit read of v is caught. */
I128 make_i128(long long k)
{
    I128 r = { ((__int128)(k + 1) << 64) | (unsigned long long)(k * 3), k };
    return r;
}

long long i128_sum(Big24 pad, I128 v)
{
    return (pad.x + 2 * pad.y + 3 * pad.z) + (long long)(v.v >> 64) * 10 +
           (long long)(unsigned long long)v.v * 100 + v.k * 1000;
}

/* The control: a natural, 8-aligned MEMORY record stays on ccall. */
long long big24_sum(Big24 v)
{
    return v.x + 2 * v.y + 3 * v.z;
}

}

long long scalar_widths(int a, unsigned int b, unsigned long long c, Kind k)
{
    return (long long)a + 1000LL * (long long)b + 1000000LL * (long long)c + 1000000000LL * (long long)k;
}
