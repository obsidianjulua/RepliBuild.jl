#ifndef NARROW_ARG_H
#define NARROW_ARG_H
#include <cstdint>

/* Library-free fixture for integer arguments narrower than `int` (2026-11-16,
   from Jolt's `BodyManager::GetNumActiveBodies(EBodyType)`, `enum class
   EBodyType : uint8`). The x86-64 convention as compilers implement it has
   the CALLER extend such an argument to 32 bits, and clang's callees rely on
   it: they read the register 32 bits wide. A thunk that passed an enum as a
   one-byte record, or a narrow integer with no extension, handed the callee
   whatever lay above the byte. Every function here uses its narrow argument
   at full register width, as an index or in 32-bit arithmetic, and is kept
   out of line so the compiler cannot see its callers. */

namespace na {

enum class Kind : uint8_t { A = 0, B = 1, C = 2 };
enum class Level : int8_t { Low = -2, Mid = 0, High = 3 };
enum Wide16 : uint16_t { W0 = 0, W9 = 9, WMax = 0xffff };
enum class Full : int { F7 = 7 };                       /* 32 bits: nothing to extend */

struct Table {
    uint32_t counts[4];
    Table();
    uint32_t count(Kind k) const;                       /* counts[k]: Jolt's shape */
    void bump(Kind k, uint8_t by);
};

uint32_t kind_index(Kind k);                            /* (uint32_t)k */
int level_value(Level l);                               /* sign-extended */
uint32_t wide16_value(Wide16 w);
int full_value(Full f);
/* Plain narrow integers, each read 32 bits wide. */
uint32_t u8_value(uint8_t v);
int s8_value(int8_t v);
uint32_t u16_value(uint16_t v);
int s16_value(short v);
int bool_value(bool b);
/* Several narrow arguments among wide ones. */
long mix(long a, Kind k, int8_t s, uint16_t u, bool b, double d, Level l);

}

#endif
