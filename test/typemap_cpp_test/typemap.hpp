// test/typemap_cpp_test: how a C++ type becomes a Julia type
// (test_type_mapping_live.jl).
//
//   - a reference and a pointer to an enum, in every position;
//   - the Unicode character types by value;
//   - an enum defined with no enumerators: the library's own, and the
//     toolchain's `std::byte`, by value and behind a pointer;
//   - `unsigned __int128` beyond the registers, which only a thunk passes.
#pragma once
#include <cstddef>
#include <cstdint>

namespace tmap {

enum class Color : std::uint8_t { Red = 1, Green = 2, Blue = 250 };
enum class Level : std::int16_t { Low = -300, Mid = 0, High = 300 };
enum class Handle : std::uint32_t {};          // no enumerators: a strong integer
enum class Opaque : int;                        // declared only: no type to emit

void   color_out(Color &c);                     // c = Blue
bool   color_is(const Color &c, Color d);
void   level_out(Level *out) noexcept;          // *out = High (a plain ccall)
int    levels_sum(const Level *xs, int n);
void   color_swap(Color &a, Color &b) noexcept; // a plain ccall with two references
Color *color_store();                           // {Green, Blue}
Color &color_first();                           // color_store()[0]

struct Palette { Color *colors; int n; Level floor; };
Palette palette_make();                         // {color_store(), 2, Low}

char8_t  u8_echo(char8_t c);
char16_t c16_echo(char16_t c);
char32_t c32_echo(char32_t c);
char16_t c16_next(char16_t c) noexcept;         // a plain ccall
// Reads each narrow argument 32 bits wide, as clang's callees do.
long     chars_mix(char8_t a, char16_t b, char32_t c, int tail);

std::byte byte_flip(std::byte b);               // ~b
int       bytes_sum(const std::byte *p, int n);
void      bytes_fill(std::byte *p, int n) noexcept;   // p[i] = i + 1
Handle    handle_next(Handle h);                // h + 1
void      opaque_set(Opaque *o) noexcept;       // the int behind it = 7

unsigned __int128 u128_echo(unsigned __int128 v) noexcept;
unsigned __int128 u128_after5(long a, long b, long c, long d, long e, unsigned __int128 v) noexcept;
unsigned __int128 u128_after7(long a, long b, long c, long d, long e, long f, long g,
                              unsigned __int128 v) noexcept;
unsigned long long u128_hi(unsigned __int128 v) noexcept;

} // namespace tmap
