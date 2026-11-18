#include "unwind_api.h"
#include <stdexcept>

/* See unwind_api.h. Every function is defined out of line, and the throwing
   ones throw on a run-time condition, so no optimization level can prove them
   `nounwind`. */

static int g_calls = 0;

extern "C" int cl_checked(int x) {
    if (x < 0) throw std::runtime_error("cl_checked: negative");
    ++g_calls;
    return x + 100;
}

extern "C" void cl_checked_void(int x) {
    if (x < 0) throw std::out_of_range("cl_checked_void: negative");
    ++g_calls;
}

extern "C" const char *cl_name(int which) {
    if (which < 0) throw std::invalid_argument("cl_name: negative");
    ++g_calls;
    return which == 0 ? "zero" : "other";
}

extern "C" ClPoint cl_point(double x, double y) {
    if (x < 0) throw std::domain_error("cl_point: negative");
    ++g_calls;
    ClPoint p = {x + 1.0, y * 2.0};
    return p;
}

extern "C" int cl_apply(int (*op)(int), int x) {
    if (x < 0) throw std::logic_error("cl_apply: negative");
    ++g_calls;
    return op(x) + 1;
}

extern "C" int cl_throws_int(int x) {
    if (x < 0) throw 42;
    ++g_calls;
    return x;
}

extern "C" int cl_add(int a, int b) { return a + b; }

extern "C" int cl_calls(void) { return g_calls; }
