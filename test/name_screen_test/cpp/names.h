// A C++ library whose records are named like the Julia types a generated
// wrapper is written in. test_name_screen_live.jl builds and wraps it.
#pragma once

#define NAME_SCREEN_RECORD(T)                    \
    struct T {                                   \
        int v;                                   \
        double w;                                \
        int get() const;                         \
    };                                           \
    T make_##T(int v);                           \
    int take_##T(const T &x);

// The six that broke a module (audit 2026-11-17, B3) ...
NAME_SCREEN_RECORD(Bool)
NAME_SCREEN_RECORD(Cvoid)
NAME_SCREEN_RECORD(Int)
NAME_SCREEN_RECORD(Integer)
NAME_SCREEN_RECORD(Cstring)
NAME_SCREEN_RECORD(UInt8)
// ... one the type sanitizer has always moved, and two the wrapper's text
// does not use.
NAME_SCREEN_RECORD(String)
NAME_SCREEN_RECORD(Task)
NAME_SCREEN_RECORD(Widget)

// C++'s own types beside the records Julia spells the same way.
bool truth(const Bool &b);
unsigned char byte_of(const UInt8 &u);
void nop(const Cvoid &c, int *out);
const char *label();
int plain(int x);

// A record holding one of them by value, passed and returned by value.
struct Holder {
    Bool inner;
    int tag;
};
Holder hold(int v);
int held(Holder h);

// A function named like every module's own binding, and a global named like
// a type of Base: its accessor is a function too.
int eval(int x);
extern int Dict;

// A class the wrapper owns (factory and deleter), whose methods are named
// like functions and a type of Base: each becomes a top-level proxy.
class File {
public:
    explicit File(int n);
    virtual ~File();
    int open();
    int isempty() const;
    int String() const;
    int get() const;
    int size() const;

private:
    int n_;
};
File *make_file(int n);
void destroy_file(File *f);
