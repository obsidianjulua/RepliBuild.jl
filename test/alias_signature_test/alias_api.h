#ifndef ALIAS_API_H
#define ALIAS_API_H

/* Library-free fixture for symbols no DIE describes (2026-11-17).
   `__attribute__((alias("f")))` gives f's code a second exported name, and
   the compiler writes debug information for f alone. The alias's signature
   was then guessed from its name (no parameters, int), and the call passed
   nothing. An alias has the signature of the function it shares an address
   with; a symbol with no debug information and no such function is refused.
   The arguments below all matter to the result, so a call that drops one
   cannot pass by luck. */

typedef struct AlPair { double a; double b; } AlPair;

int    al_real(int x);                        /* x + 1 */
int    al_name(int x);                        /* alias of al_real */
double al_scaled_real(double x, int n);       /* x * n */
double al_scaled(double x, int n);            /* weak alias of al_scaled_real */
AlPair al_pair_real(double a, double b);      /* {a + b, a - b} */
AlPair al_pair(double a, double b);           /* alias: a record by value */
long   al_wide_real(long a, long b, long c, long d, long e, long f, long g);
long   al_wide(long a, long b, long c, long d, long e, long f, long g);   /* alias: a stack argument */
int    al_old_style();                        /* no prototype; 1978 */
int    al_undescribed(int x);                 /* no debug information: x + 2 */

#endif
