// A second translation unit that defines one member of Bird and nothing else.
// Under clang's default debug info the unit with a class's constructor
// describes the class; this one holds Bird as a declaration that lists `speed`
// alone. A class table that keeps the last DIE of a name keeps this one
// wherever this unit is linked after vclass.cpp.
#include "vclass.h"

double Bird::speed(double terrain) const { return 30.0 * terrain + wingspan; }
