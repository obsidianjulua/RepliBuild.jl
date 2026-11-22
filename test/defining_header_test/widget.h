#pragma once
// A clean header: everything in it is inline.
struct Widget {
    int v;
    Widget(int x) : v(x) {}                // constructed by no unit of the library
    int twice() const { return 2 * v; }    // called by none
};
