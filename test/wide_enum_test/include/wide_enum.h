#ifndef WIDE_ENUM_H
#define WIDE_ENUM_H
#include <cstdint>

/* Library-free fixture for an enumerator above typemax(Int64) (2026-11-16,
   from Jolt's LargeIslandSplitter::EIterationStatus : uint64, whose
   0xffff000000000000 made the bitflag test throw and ended the whole wrap). */

/* Not a flag set: masks and shifts side by side, as Jolt's is. */
enum Status : uint64_t {
    StatusIterationMask = 0xffff000000000000ull,
    StatusIterationShift = 48,
    StatusSplitMask = 0x0000ffff00000000ull,
    StatusSplitShift = 32,
    StatusItemMask = 0x00000000ffffffffull,
};

/* A flag set whose top bit is bit 63. */
enum Wide : uint64_t {
    WideA = 1ull << 60,
    WideB = 1ull << 61,
    WideC = 1ull << 62,
    WideTop = 1ull << 63,
};

/* An ordinary flag set, as before. */
enum Small : unsigned { SmallA = 1, SmallB = 2, SmallC = 4 };

struct Holder {
    Status s;
    Wide w;
};

uint64_t status_bits(Status s);
Status status_mask();
int wide_top_set(Wide w);
Wide wide_all();
unsigned small_bits(Small s);

#endif
