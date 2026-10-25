/* Live records for test_sysv_live.jl: one per way a by-value record can cross
   a Tier 2 thunk under x86-64 SysV, with the records classified from their
   DWARF facts (FunctionGen `sysv_record_abi`), and the controls that always
   agreed. take_R / make_R have C++ linkage and are not noexcept, so a C++
   wrap routes them to Tier 2 thunks: the classification under test is the
   dialect's. image_R / image_mask_R / size_R are extern "C" with pointer or
   no arguments: Tier 3, trusted. */
#include <stddef.h>
#include <stdint.h>

/* an eightbyte holding a float and padding is SSE */
struct c_f_d    { float x; double y; };
struct c_d_f    { double d; float f; };
struct c_p_f    { void *p; float f; };
struct c_ubf_f  { int : 32; float f; };
struct c_f_pad  { float x; int : 32; double y; };
struct c_wrap_df { c_d_f in; };
struct c_al16_f { float f __attribute__((aligned(16))); };
struct c_cf     { _Complex float z; };
struct c_cd     { _Complex double z; };
/* a low eightbyte of padding is passed nowhere; the high one travels alone */
struct c_ubf_d  { long long : 64; double d; };
/* long double: an argument in memory (byval 16), a return in st0 (a trap) */
struct c_ld     { long double x; };
union  c_u_ld_i { long double x; int i; };
/* misaligned fields make a record MEMORY, through a nested record or a union */
struct c_s_f    { short s; float f; };
struct __attribute__((packed)) c_pk_sf { char c; c_s_f in; };
struct __attribute__((packed)) c_pk_ci { char c; int i; };
union  c_u_pk   { c_pk_ci p; int i; };
struct __attribute__((aligned(8))) c_al8_c { char x; };
#pragma pack(push, 1)
struct c_pk_alnest { char c; c_al8_c s; };      /* the nested record itself is misaligned; its one char is not */
struct c_pp1_bf { char c; int b : 31; };        /* a bitfield classifies by its bits: one register */
#pragma pack(pop)
/* a union and a bitfield record of 16 bytes or less, classified rather than refused */
union  c_u_fi   { float f; int i; };
struct c_bf2_f  { int a : 3; int b : 5; float f; };
/* 16-aligned records: byval and sret slots of 16 */
struct c_i128_l { __int128 x; long long y; };
struct __attribute__((aligned(16))) c_al16_big { long long a, b, c; };
/* controls */
struct c_f_i    { float x; int i; };
struct c_d2     { double a, b; };
struct c_c_d    { char c; double d; };

extern "C" void poison_regs();
#define DECL(R) \
    double take_##R(R v); \
    R make_##R(); \
    extern "C" void image_##R(void *out); \
    extern "C" void image_mask_##R(void *out); \
    extern "C" unsigned size_##R();

DECL(c_f_d) DECL(c_d_f) DECL(c_p_f) DECL(c_ubf_f) DECL(c_f_pad) DECL(c_wrap_df) DECL(c_al16_f)
DECL(c_cf) DECL(c_cd) DECL(c_ubf_d) DECL(c_ld) DECL(c_u_ld_i) DECL(c_s_f) DECL(c_pk_sf) DECL(c_pk_ci)
DECL(c_u_pk) DECL(c_pk_alnest) DECL(c_pp1_bf) DECL(c_u_fi) DECL(c_bf2_f) DECL(c_i128_l) DECL(c_al16_big)
DECL(c_f_i) DECL(c_d2) DECL(c_c_d)
