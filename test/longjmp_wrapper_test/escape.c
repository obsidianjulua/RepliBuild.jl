/* test_longjmp_wrappers_live.jl: C functions that leave by longjmp, called
   from a Julia callback. This is how a C callback raises an error in Lua
   (luaL_error), libpng (png_error) and libjpeg (error_exit): the library sets
   a jump point, calls back, and the callback calls a function that never
   returns to it. The Julia frames between the two are abandoned, so a
   wrapper on that path must hold nothing the collector has to be told about. */
#include <setjmp.h>
#include <stdarg.h>
#include <string.h>

typedef int (*lj_callback)(void *);

static jmp_buf lj_env;
static char lj_message[64];
static int lj_code;

static void lj_note(const char *message) {
    strncpy(lj_message, message, sizeof lj_message - 1);
    lj_message[sizeof lj_message - 1] = 0;
}

/* Leaves by longjmp with a message. */
void lj_fail(const char *message) {
    lj_note(message);
    lj_code = 0;
    longjmp(lj_env, 1);
}

/* The variadic form (luaL_error's shape): the fixed parameter is the text,
   the first variadic argument an int kept as the code. */
int lj_failf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    lj_code = va_arg(ap, int);
    va_end(ap);
    lj_note(fmt);
    longjmp(lj_env, 1);
    return 0;
}

/* Calls f. 1 when f left through lj_fail or lj_failf, 0 when it returned. */
int lj_protected(lj_callback f, void *data) {
    if (setjmp(lj_env) != 0) return 1;
    f(data);
    return 0;
}

const char *lj_last_message(void) { return lj_message; }
int lj_last_code(void) { return lj_code; }

/* A text that returns normally, for the same wrapper shape off the jump. */
size_t lj_len(const char *s) { return strlen(s); }

/* Bytes passed with their length: a NUL among them is data. */
size_t lj_count_nul(const char *bytes, size_t n) {
    size_t k = 0;
    for (size_t i = 0; i < n; i++) k += bytes[i] == 0;
    return k;
}
