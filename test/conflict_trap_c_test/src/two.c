/* The other `struct state` (see one.c): the same size, another layout. */

struct state { double v; };

double st_two(struct state s) { return s.v * 2.0; }
