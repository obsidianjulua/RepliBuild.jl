#ifndef VISIBLE_RECORD_H
#define VISIBLE_RECORD_H

/* Library-free fixture for a header-only class that no source of the library
   mentions (2026-11-15, from Bullet3's btDefaultMotionState). Debug info
   describes the types a translation unit uses, so such a class is in no
   unit's DWARF, with -fstandalone-debug or without: the wrapper had no type
   for it, and the build's inline-constructor pass had no declaration to
   force. `[types] records` names it, and a generated unit uses it once
   (Compiler.generate_visible_records). */

namespace vr {

/* Defined in the .cpp and used by the library. */
struct Used {
    Used(int v);
    int v;
};

/* Header-only, used by nothing in the library: invisible until listed. */
struct Unused {
    Unused(int x) : v(x * 2) {}
    int twice() const { return v * 2; }
    int v;
};

/* A second one, nested, to check a qualified spelling. */
struct Outer {
    struct Inner {
        Inner(float f) : f(f) {}
        float f;
    };
};

/* Only declared: listing it cannot work, and the build must say so. */
struct Incomplete;

int used_value(const Used *u);
/* Read through `const void*` on purpose: a parameter of type `Unused*` would
   itself put the class in the DWARF. */
int unused_value(const void *unused);
float inner_value(const void *inner);

}

#endif
