#include "names.h"

int String(int x) { return x + 101; }
int Ptr(int x) { return x + 102; }
int Ref(int x) { return x + 103; }
int Cint(int x) { return x + 104; }
int eval(int x) { return x + 105; }
int include(int x) { return x + 106; }
int ccall(int x) { return x + 107; }
int Base(int x) { return x + 108; }
int Libdl(int x) { return x + 109; }
int RepliBuild(int x) { return x + 110; }
int nothing(int x) { return x + 111; }
int unsafe_convert(int x) { return x + 112; }
int isfile(int x) { return x + 113; }
int LIBRARY_PATH(int x) { return x + 114; }
int METADATA(int x) { return x + 115; }
int Namescreenc(int x) { return x + 116; }
int length(int x) { return x + 117; }
int print(int x) { return x + 118; }
int show(int x) { return x + 119; }
int plain(int x) { return x + 1; }
const char *label(void) { return "lbl"; }

Integer make_integer(int v) { Integer r = {v, v * 0.5}; return r; }
int take_integer(const Integer *x) { return x->v + 1; }
Bool make_bool(int v) { Bool r = {v, v * 0.5}; return r; }
int take_bool(const Bool *x) { return x->v + 2; }
Task make_task(int v) { Task r = {v, v * 0.5}; return r; }
int take_task(const Task *x) { return x->v + 3; }
bool truth(int x) { return x > 0; }

enum Kind kind_of(int x) { return (enum Kind)x; }

int Dict = 7;
int counter = 9;
