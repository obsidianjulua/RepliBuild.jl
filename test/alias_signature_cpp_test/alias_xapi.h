#ifndef ALIAS_XAPI_H
#define ALIAS_XAPI_H

/* The C++ half of test/alias_signature_test (see alias_api.h there): aliases
   with C linkage and with a mangled name, two functions with no debug
   information at all, and an alias whose aliasee is a `static` function. That
   one is refused as well: its aliasee is a local symbol, and a local name
   need not be one function's (two files may each have a `static` of that
   name), so nothing ties it to one description. */

extern "C" {
int ax_real(int x);                 /* x + 1 */
int ax_name(int x);                 /* alias of ax_real */
int ax_via_static(int a, int b);    /* alias of a static function: refused */
int ax_undescribed(int x);          /* no debug information: x + 2 */
}

double ax_mangled_real(double x, int n);   /* x * n */
double ax_mangled(double x, int n);        /* alias of ax_mangled_real, mangled itself */
int    ax_undescribed_cxx(int x);          /* no debug information, mangled: x + 3 */

#endif
