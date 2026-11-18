/* A C library whose identifiers are names Julia, or a generated wrapper,
 * already uses. test_name_screen_live.jl builds and wraps it; every function
 * returns its argument plus a number of its own, so the probe can tell which
 * Julia name reaches which C function. */
#ifndef NAME_SCREEN_C_H
#define NAME_SCREEN_C_H

#include <stdbool.h>

/* Types of Base and Core. */
int String(int x);
int Ptr(int x);
int Ref(int x);
int Cint(int x);
/* Bindings every module has, and a name Julia's syntax keeps. */
int eval(int x);
int include(int x);
int ccall(int x);
/* Names the wrapper's own text uses or binds. */
int Base(int x);
int Libdl(int x);
int RepliBuild(int x);
int nothing(int x);
int unsafe_convert(int x);
int isfile(int x);
int LIBRARY_PATH(int x);
int METADATA(int x);
/* The module's own name (project `namescreenc`). */
int Namescreenc(int x);
/* Base functions the wrapper's text does not use: they keep their names. */
int length(int x);
int print(int x);
int show(int x);
/* And ordinary ones. */
int plain(int x);
const char *label(void);

/* Records: two names every wrapper's text uses, one it does not. */
typedef struct Integer { int v; double w; } Integer;
typedef struct Bool { int v; double w; } Bool;
typedef struct Task { int v; double w; } Task;
Integer make_integer(int v);
int take_integer(const Integer *x);
Bool make_bool(int v);
int take_bool(const Bool *x);
Task make_task(int v);
int take_task(const Task *x);
/* C's own bool beside the record of that name. */
bool truth(int x);

enum Kind { Symbol = 1, stat = 2, KindPlain = 3 };
enum Kind kind_of(int x);

extern int Dict;
extern int counter;

#endif
