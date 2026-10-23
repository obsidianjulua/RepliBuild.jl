/* Types a C unit and the C++ units both define, the way ggml.h's are. C spells
 * `bool` as `_Bool`, and the union member is anonymous in C and named in C++
 * (ggml-common.h's GGML_COMMON_AGGR_U). The C unit's DWARF and the C++ units'
 * are separate definitions of each type under one name. */
#ifndef DT_SHARED_H
#define DT_SHARED_H

#ifdef __cplusplus
#  define DT_AGGR_U data
extern "C" {
#else
#  include <stdbool.h>
#  define DT_AGGR_U
#endif

typedef struct dt_traits {
    int size;
    bool is_quantized;
} dt_traits;

typedef struct dt_block {
    union {
        int dm;
        unsigned short half[2];
    } DT_AGGR_U;
    int qs;
} dt_block;

/* A callback parameter declared with a typedef: its docstring example must be
 * this signature and no other. */
typedef int (*dt_progress_cb)(int done, void *user);

/* A decoy. Its name is inside `dt_run_with`'s and `dt_run_raw`'s, which is what
 * the old name-similarity scoring rewarded over the parameter's own typedef. */
typedef double (*dt_run)(double x);

int dt_run_with(dt_progress_cb cb, void *user);   /* C unit */
int dt_run_raw(int (*cb)(int));                   /* C unit: no typedef, no example */

/* Globals, emitted as accessors in name order. */
extern int dt_g_zeta;
extern int dt_g_alpha;
extern int dt_g_mid;
bool dt_traits_quantized(const dt_traits *t);     /* C unit */
int dt_block_qs(const dt_block *b);               /* C unit */
int dt_traits_size(const dt_traits *t);           /* C++ unit */
int dt_block_dm(const dt_block *b);               /* C++ unit */

#ifdef __cplusplus
}
#endif

#endif
