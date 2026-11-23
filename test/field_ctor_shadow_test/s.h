/* A function named like a struct, with one parameter per field
 * (test_field_ctor_shadow_live.jl). C keeps struct tags apart from ordinary
 * names and C++ lets a function hide a class name, so `pair` is two names in
 * either; in Julia it is one, and the function took the calls of the
 * struct's field constructor `pair(a, b)`. The others must keep their names:
 * a bitfield record, a union and a packed struct are built from their bytes,
 * and `triple(int)` has another arity than its struct. */
#ifndef FIELD_CTOR_SHADOW_S_H
#define FIELD_CTOR_SHADOW_S_H

#ifdef __cplusplus
#  define RB_NOEXCEPT noexcept
#else
#  define RB_NOEXCEPT
#endif

struct pair { int a; int b; };
int pair(int a, int b) RB_NOEXCEPT;
int pair_sum(struct pair p) RB_NOEXCEPT;
struct pair make_pair(int a, int b) RB_NOEXCEPT;

struct triple { int x; int y; int z; };
int triple(int x) RB_NOEXCEPT;
int triple_sum(const struct triple *t) RB_NOEXCEPT;

struct flags { unsigned lo : 3; unsigned hi : 5; };
int flags(int lo, int hi) RB_NOEXCEPT;
struct flags make_flags(void) RB_NOEXCEPT;

union num { int i; float f; };
int num(int i) RB_NOEXCEPT;
union num make_num(void) RB_NOEXCEPT;

struct packed_s { char c; int v; } __attribute__((packed));
int packed_s(int c, int v) RB_NOEXCEPT;
struct packed_s make_packed(void) RB_NOEXCEPT;

#endif
