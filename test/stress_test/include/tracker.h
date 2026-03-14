#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct Tracker {
    int constructed;
    int destructed;
};

void tracker_ctor(struct Tracker* t);
void tracker_dtor(struct Tracker* t);
void tracker_ctor_val(struct Tracker* t, int val);

#ifdef __cplusplus
}
#endif
