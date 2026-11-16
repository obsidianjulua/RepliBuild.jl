#include "visible_record.h"

namespace vr {

Used::Used(int v) : v(v) {}
int used_value(const Used *u) { return u->v; }
/* The first member of each is its only field; read as raw storage so this
   file does not use the class. */
int unused_value(const void *unused) { return *static_cast<const int *>(unused); }
float inner_value(const void *inner) { return *static_cast<const float *>(inner); }

}
