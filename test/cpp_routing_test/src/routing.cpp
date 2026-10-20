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

}
