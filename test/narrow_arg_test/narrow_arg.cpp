#include "narrow_arg.h"

namespace na {

Table::Table() : counts{10, 20, 30, 40} {}
uint32_t Table::count(Kind k) const { return counts[(uint32_t)k]; }
void Table::bump(Kind k, uint8_t by) { counts[(uint32_t)k] += by; }

uint32_t kind_index(Kind k) { return (uint32_t)k; }
int level_value(Level l) { return (int)l; }
uint32_t wide16_value(Wide16 w) { return (uint32_t)w; }
int full_value(Full f) { return (int)f; }
uint32_t u8_value(uint8_t v) { return v; }
int s8_value(int8_t v) { return v; }
uint32_t u16_value(uint16_t v) { return v; }
int s16_value(short v) { return v; }
int bool_value(bool b) { return b ? 100 : 200; }
long mix(long a, Kind k, int8_t s, uint16_t u, bool b, double d, Level l) {
    return a + 1000 * (long)(uint32_t)k + 100 * (long)s + (long)u + (b ? 7 : 0) + (long)d + 10000 * (long)(int)l;
}

}
