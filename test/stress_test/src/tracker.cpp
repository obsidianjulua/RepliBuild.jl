#include "tracker.h"

extern "C" {

void tracker_ctor(Tracker* t) {
    t->constructed = 42;
}

void tracker_dtor(Tracker* t) {
    t->destructed = 99;
}

void tracker_ctor_val(Tracker* t, int val) {
    t->constructed = val;
}

}
