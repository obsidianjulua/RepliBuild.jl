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

}
