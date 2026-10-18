#include "c_sret.h"

unsigned long long c1_rec[16];
int c1_rec_n;

/* What the callee read, widened by its own signedness. */
#define REC(i, e) (c1_rec[i] = (unsigned long long)(long long)(e))

T4 f4(R5 p1)
{
    REC(0, p1.m1);
    c1_rec_n = 1;
    T4 r = { 1, { 22018, 29443, 23812 } };
    return r;
}

T4 f2(T4 p1, R4 p3)
{
    REC(0, p1.m1);
    REC(1, p1.m2[0]);
    REC(2, p1.m2[1]);
    REC(3, p1.m2[2]);
    REC(4, p3.m1);
    REC(5, p3.m2);
    c1_rec_n = 6;
    T4 r = { 1, { 11778, -26365, 11268 } };
    return r;
}

IC12 make_ic12(int n, char c)
{
    REC(0, n);
    REC(1, c);
    c1_rec_n = 2;
    IC12 r;
    r.n = n * 3;
    for (int i = 0; i < 12; i++)
        r.s[i] = (char)(c + i);
    return r;
}

int ic12_sum(IC12 v)
{
    int s = v.n;
    for (int i = 0; i < 12; i++)
        s += v.s[i];
    return s;
}

P4 make_p4(short s)
{
    REC(0, s);
    c1_rec_n = 1;
    P4 r = { 0x11, (short)(s + 1), 0x22 };
    return r;
}

P8 make_p8(int i)
{
    REC(0, i);
    c1_rec_n = 1;
    P8 r = { 0x33, i ^ 0x5a5a5a5a, { 7, 8, 9 } };
    return r;
}

int take_p8(P8 v)
{
    REC(0, v.c);
    REC(1, v.i);
    REC(2, v.pad[0]);
    REC(3, v.pad[1]);
    REC(4, v.pad[2]);
    c1_rec_n = 5;
    return v.i + v.c;
}

Big24 make_big24(long long a, long long b, int c)
{
    REC(0, a);
    REC(1, b);
    REC(2, c);
    c1_rec_n = 3;
    Big24 r = { a + 1, b + 2, c + 3 };
    return r;
}

IS20 make_is20(int n)
{
    REC(0, n);
    c1_rec_n = 1;
    IS20 r;
    r.n = n + 1;
    for (int i = 0; i < 8; i++)
        r.s[i] = (short)(n * (i + 1));
    return r;
}

PK17 make_pk17(long long x)
{
    REC(0, x);
    c1_rec_n = 1;
    PK17 r = { 0x55, x + 1, x - 1 };
    return r;
}

Out6 make_out6(short y)
{
    REC(0, y);
    c1_rec_n = 1;
    Out6 r = { { 0x44, (short)(y + 5) }, { 1, 2, 3 } };
    return r;
}

BF12 make_bf12(float f, int b, int c)
{
    union { float f; unsigned int u; } bits;
    bits.f = f;
    c1_rec[0] = bits.u;
    REC(1, b);
    REC(2, c);
    c1_rec_n = 3;
    BF12 r = { f * 2.0f, b, c + 1 };
    return r;
}

/* Over-aligned records: each maker fills the aligned global from its arguments
   and returns it, so the copy into the hidden return buffer is an aligned
   16-byte store. */
AB16 g_ab16;
V4A g_v4a;
D4A g_d4a;
MA32 g_ma32;
A32 g_a32;
HVec g_hvec;
HV4 g_hv4;
NestA g_nesta;

AB16 make_ab16(float a, int c)
{
    union { float f; unsigned int u; } bits;
    bits.f = a;
    c1_rec[0] = bits.u;
    REC(1, c);
    c1_rec_n = 2;
    g_ab16.a = a;
    g_ab16.b = a * 2.0f;
    g_ab16.c = c;
    return g_ab16;
}

V4A make_v4a(float x)
{
    union { float f; unsigned int u; } bits;
    bits.f = x;
    c1_rec[0] = bits.u;
    c1_rec_n = 1;
    g_v4a.x = x;
    g_v4a.y = x + 1.0f;
    g_v4a.z = x + 2.0f;
    g_v4a.w = x + 3.0f;
    return g_v4a;
}

D4A make_d4a(double a)
{
    union { double f; unsigned long long u; } bits;
    bits.f = a;
    c1_rec[0] = bits.u;
    c1_rec_n = 1;
    g_d4a.a = a;
    g_d4a.b = -a;
    g_d4a.c = a * 2.0;
    g_d4a.d = a / 2.0;
    return g_d4a;
}

MA32 make_ma32(char c, int i)
{
    REC(0, c);
    REC(1, i);
    c1_rec_n = 2;
    g_ma32.c = c;
    g_ma32.i = i * 3;
    return g_ma32;
}

A32 make_a32(double d)
{
    union { double f; unsigned long long u; } bits;
    bits.f = d;
    c1_rec[0] = bits.u;
    c1_rec_n = 1;
    g_a32.d[0] = d;
    g_a32.d[1] = -d;
    return g_a32;
}

HVec make_hvec(float v, int k)
{
    union { float f; unsigned int u; } bits;
    bits.f = v;
    c1_rec[0] = bits.u;
    REC(1, k);
    c1_rec_n = 2;
    for (int i = 0; i < 4; i++)
        g_hvec.v[i] = v + (float)i;
    g_hvec.k = k + 1;
    return g_hvec;
}

HV4 make_hv4(float v, float s)
{
    union { float f; unsigned int u; } bits;
    bits.f = v;
    c1_rec[0] = bits.u;
    bits.f = s;
    c1_rec[1] = bits.u;
    c1_rec_n = 2;
    for (int i = 0; i < 4; i++)
        g_hv4.v[i] = v * (float)(i + 1);
    g_hv4.s = s;
    return g_hv4;
}

NestA make_nesta(float x, float t)
{
    union { float f; unsigned int u; } bits;
    bits.f = x;
    c1_rec[0] = bits.u;
    bits.f = t;
    c1_rec[1] = bits.u;
    c1_rec_n = 2;
    g_nesta.a.x = x;
    g_nesta.a.y = -x;
    g_nesta.a.z = x * 4.0f;
    g_nesta.a.w = 0.25f;
    g_nesta.t = t;
    return g_nesta;
}
