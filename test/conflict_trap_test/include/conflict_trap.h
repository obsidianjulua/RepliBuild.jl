#ifndef CONFLICT_TRAP_H
#define CONFLICT_TRAP_H

/* Library-free fixture for a bare type name several types share, as
   llama.cpp's pimpl `impl`s do. RepliBuild keys types by their bare DWARF
   name and keeps one definition per name, so a function that takes or returns
   one of them by value, or a record holding one, has no Julia layout it can
   trust: each of those is a trap. An enum crosses as an integer of its width,
   so `Mode` (4 bytes in both namespaces) is still called; `Kind` (1 byte and
   4 bytes) is not. */

namespace alpha {
struct impl { int n_layer; int n_head; };                   /* 8 B */
enum Mode { MODE_X = 0, MODE_Y = 1 };                       /* 4 B */
enum class Kind : unsigned char { K_A = 0, K_B = 1 };       /* 1 B */
}

namespace beta {
struct impl { double scores; int n_vocab; };                /* 16 B */
enum Mode { MODE_P = 0, MODE_Q = 1, MODE_R = 2 };           /* 4 B */
enum class Kind : int { K_P = 0, K_Q = 7 };                 /* 4 B */
}

struct Duo { int a; int b; };                               /* one type */
struct Holder { alpha::impl i; int k; };                    /* holds alpha::impl */

extern "C" {
/* C linkage, so exception routing does not apply: these would be Tier 3. */
int ct_impl_sum(alpha::impl v);                  /* trap */
int ct_holder_sum(Holder h);                     /* trap: holds `impl` */
int ct_alpha_kind(alpha::Kind k);                /* trap: `Kind` is 1 B and 4 B */
int ct_duo_sum(Duo d);                           /* Tier 3 */
int ct_mode_value(alpha::Mode m);                /* Tier 3: every `Mode` is 4 B */
int ct_impl_ptr_sum(const alpha::impl *v);       /* Tier 3: by address */
}

/* C++ linkage and may throw: Tier 2. */
double ct_vocab_score(beta::impl v);             /* trap */
beta::Kind ct_kind(int x);                       /* trap */
int ct_duo_twice(Duo d);                         /* Tier 2 */
int ct_beta_mode(beta::Mode m);                  /* Tier 2 */

#endif
