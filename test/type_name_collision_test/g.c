#include "g.h"

struct conf conf_default = { 3, 1 };
const struct conf *conf = &conf_default;

int conf_level(const struct conf *c) { return c->level; }

static struct pool the_pool = { 5, 2 };
struct pool *pool(void) { return &the_pool; }

struct token token(void) { struct token t = { 7, 2 }; return t; }

int counter = 9;
