// test_const_members_live.jl: records whose members are const-qualified,
// returned and passed by value. The shape is llama.cpp's LLM_TN_IMPL: five
// const members that tile 24 bytes with no padding, returned by value from a
// member function.
#include <cstring>

enum Kind { KIND_A = 1, KIND_B = 2, KIND_C = 7 };

// 4 + 4 + 8 + 4 + 4 = 24 bytes, no padding: MEMORY class on SysV.
struct Name {
    const Kind arch;
    const Kind tensor;
    const char *const suffix;
    const int bid;
    const int xid;
};

// 4 + 4 = 8 bytes, no padding: one INTEGER register.
struct Pair {
    const int a;
    const unsigned b;
};

// Padding after the first member, a volatile member, a const double.
struct Mixed {
    const char tag;
    volatile int count;
    const double scale;
};

struct Namer {
    Kind arch;
    Name make(Kind tensor, const char *suffix, int bid, int xid) const;
    Name operator()(Kind tensor, int bid, int xid) const;
};

Name Namer::make(Kind tensor, const char *suffix, int bid, int xid) const {
    return Name{arch, tensor, suffix, bid, xid};
}
Name Namer::operator()(Kind tensor, int bid, int xid) const {
    return Name{arch, tensor, nullptr, bid, xid};
}

Pair pair_of(int a, unsigned b) { return Pair{a, b}; }
long pair_sum(Pair p) { return static_cast<long>(p.a) + p.b; }

Mixed mixed_of(char tag, int count, double scale) { return Mixed{tag, count, scale}; }
double mixed_total(Mixed m) { return m.tag + m.count + m.scale; }

long name_total(Name n) {
    return n.arch * 1000000L + n.tensor * 10000L + n.bid * 100L + n.xid +
           (n.suffix ? static_cast<long>(std::strlen(n.suffix)) * 100000000L : 0);
}

// A member the thunk generator has no type for (a bit-precise integer):
// the metadata tiles 4 + 4 + 8 + 8 = 24 bytes, and the record's body is a
// byte region. Over 16 bytes it is MEMORY class whatever its members are, so
// it gets a thunk; it has to cross as its bytes. Marshalled field by field,
// the module does not verify and every thunk of the library is lost.
struct Wide {
    unsigned _BitInt(24) bits;
    int n;
    long long a;
    long long b;
};
Wide wide_of(int bits, int n, long long a, long long b) {
    return Wide{static_cast<unsigned _BitInt(24)>(bits), n, a, b};
}
long long wide_sum(Wide w) { return static_cast<long long>(w.bits) + w.n + w.a + w.b; }

// Scalars qualified in a signature, by value. A definition's DWARF keeps the
// qualifier (`const double x`), and the value still travels as a double.
double scaled(const double x, const int n) { return x * n; }
float halved(const float x) { return x / 2; }
const double doubled(const double x) { return x + x; }
// Narrower than int: extended by the caller, by the type's own signedness.
int narrow(const unsigned char a, const signed char b, const short c, const bool d) {
    return a + b + c + (d ? 1 : 0);
}
long long mixed_args(const int a, const double b, const long long c, const float d) {
    return a + static_cast<long long>(b) + c + static_cast<long long>(d);
}

// A function that may throw, so the library has a thunk on the EH path too.
int checked(int x) {
    if (x < 0) throw x;
    return x + 1;
}
