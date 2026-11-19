/* Identifiers that are a struct's tag and an ordinary name at once. C keeps
 * tags in a namespace of their own, so each pair is two names in C and one in
 * Julia (test_type_name_collisions_live.jl). */
#ifndef TYPE_NAME_COLLISION_G_H
#define TYPE_NAME_COLLISION_G_H

/* A global named like its struct tag: curl's `Curl_ssl`. */
struct conf { int level; int flags; };
extern const struct conf *conf;
extern struct conf conf_default;
int conf_level(const struct conf *c);

/* A function taking nothing, named like the struct it points at. */
struct pool { int size; int used; };
struct pool *pool(void);

/* A function taking nothing, returning its struct by value. */
struct token { int kind; int len; };
struct token token(void);

/* Nothing collides. */
extern int counter;

#endif
