#include "names.h"

#define NAME_SCREEN_RECORD_IMPL(T, K)                              \
    int T::get() const { return v + K; }                           \
    T make_##T(int v) { return T{v, v * 0.5}; }                    \
    int take_##T(const T &x) { return x.v + static_cast<int>(x.w) + K; }

NAME_SCREEN_RECORD_IMPL(Bool, 10)
NAME_SCREEN_RECORD_IMPL(Cvoid, 20)
NAME_SCREEN_RECORD_IMPL(Int, 30)
NAME_SCREEN_RECORD_IMPL(Integer, 40)
NAME_SCREEN_RECORD_IMPL(Cstring, 50)
NAME_SCREEN_RECORD_IMPL(UInt8, 60)
NAME_SCREEN_RECORD_IMPL(String, 70)
NAME_SCREEN_RECORD_IMPL(Task, 80)
NAME_SCREEN_RECORD_IMPL(Widget, 90)

bool truth(const Bool &b) { return b.v > 0; }
unsigned char byte_of(const UInt8 &u) { return static_cast<unsigned char>(u.v); }
void nop(const Cvoid &c, int *out) { *out = c.v + 7; }
const char *label() { return "lbl"; }
int plain(int x) { return x + 1; }

Holder hold(int v) { return Holder{Bool{v, v * 2.0}, v + 10}; }
int held(Holder h) { return h.inner.v + static_cast<int>(h.inner.w) + h.tag; }

int eval(int x) { return x + 105; }
int Dict = 11;

File::File(int n) : n_(n) {}
File::~File() {}
int File::open() { return n_ + 1; }
int File::isempty() const { return n_ == 0; }
int File::String() const { return n_ + 2; }
int File::get() const { return n_ + 3; }
int File::size() const { return n_ + 4; }
File *make_file(int n) { return new File(n); }
void destroy_file(File *f) { delete f; }
