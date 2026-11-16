#ifndef NESTED_TYPE_H
#define NESTED_TYPE_H

/* Library-free fixture for types declared inside a class (2026-11-15, from
   Bullet3's btCollisionWorld::ClosestRayResultCallback). DWARF keys a record
   by its bare name, so the struct is `RayResult` while its functions are
   `World_RayResult_…`. The wrapper now also binds the type under the name
   its functions carry (`World_RayResult`), and keys its layout by both. */

struct World {
    /* Nested in a class at global scope, as Bullet's are. */
    struct RayResult {
        RayResult(float fraction, int id);
        bool hit() const;
        float fraction;
        int id;
    };

    /* Nested two deep. */
    struct Callback {
        struct Info {
            Info(int code);
            int code;
            int pad;
        };
        int n;
    };

    /* `World_Mark` is also a free function below: the alias would rebind a
       name the module already has, so this type keeps its bare name alone. */
    struct Mark {
        Mark(int m);
        int m;
    };

    int cast(const RayResult &r) const;
    int bodies;
};

namespace nt {

/* In a namespace only: not class-nested, no alias. */
struct Plain {
    Plain(int v);
    int v;
};

/* Nested in a class inside a namespace: the alias carries both. */
struct Box {
    struct Lid {
        Lid(int w);
        int w;
    };
    int n;
};

}

int World_Mark(const World::Mark *m);
int info_code(const World::Callback::Info *i);
int lid_w(const nt::Box::Lid *l);
int plain_v(const nt::Plain *p);
int box_n(const nt::Box *b);
int callback_n(const World::Callback *c);

#endif
