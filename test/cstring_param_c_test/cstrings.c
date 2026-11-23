/* test_cstring_params_live.jl: pointers to `char` through a C wrapper (every
   function a ccall). A `const char*` parameter takes a Julia string with
   `Cstring` semantics; only a pointer to plain `char` is a C string. */
#include <stdarg.h>
#include <stddef.h>

size_t cs_len(const char *s) {
    size_t n = 0;
    while (s[n]) n++;
    return n;
}

/* A buffer the library writes into, then a string it reads. */
size_t cs_copy(char *dst, size_t cap, const char *src) {
    size_t n = 0;
    while (src[n] && n + 1 < cap) { dst[n] = src[n]; n++; }
    if (cap) dst[n] = 0;
    return n;
}

/* Bytes, not text. */
size_t cs_sum_ubytes(const unsigned char *b, size_t n) {
    size_t s = 0;
    for (size_t i = 0; i < n; i++) s += b[i];
    return s;
}

/* Text returned. */
const char *cs_greet(void) { return "hello"; }

/* `signed char*` and `unsigned char*` returned: bytes, not text. */
static const signed char g_sbytes[] = {-1, -2, 0};
static const unsigned char g_ubytes[] = {1, 2, 3, 0};
const signed char *cs_sbytes(void) { return g_sbytes; }
const unsigned char *cs_ubytes(void) { return g_ubytes; }

/* A variadic function: the fixed `fmt` is a C string. Returns its length plus
   the `count` ints that follow. */
int cs_fmt_len(const char *fmt, int count, ...) {
    int n = 0;
    while (fmt[n]) n++;
    va_list ap;
    va_start(ap, count);
    for (int i = 0; i < count; i++) n += va_arg(ap, int);
    va_end(ap);
    return n;
}

/* A `char*` that is not `const`: the function writes into it. A Julia
   string is refused there; a byte buffer is written. */
void cs_upcase(char *s) {
    for (; *s; s++)
        if (*s >= 'a' && *s <= 'z') *s = (char)(*s - 'a' + 'A');
}
