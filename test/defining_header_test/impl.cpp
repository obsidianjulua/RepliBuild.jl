// The one unit meant to include the two defining headers.
#include "exports.h"
#include "impl.h"
#include "shape.h"
#include "widget.h"

int widget_value(const Widget *w) { return w->v; }
int gadget_value(const Gadget *x) { return x->g; }
double shape_width(const Shape *s) { return s->w; }
