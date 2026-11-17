#include "wide_enum.h"

uint64_t status_bits(Status s) { return (uint64_t)s; }
Status status_mask() { return StatusIterationMask; }
int wide_top_set(Wide w) { return (w & WideTop) != 0; }
Wide wide_all() { return (Wide)(WideA | WideB | WideC | WideTop); }
unsigned small_bits(Small s) { return (unsigned)s; }
