/* Library-free fixture: each of one.c and two.c defines its own `struct state`,
   which C allows, so the name `state` stands for two types and neither
   function that takes one by value can be called. `struct pt` is one type. */

struct state { int a; int b; };

int st_one(struct state s) { return s.a * 10 + s.b; }

struct pt { int x; int y; };

int pt_sum(struct pt p) { return p.x + 2 * p.y; }
