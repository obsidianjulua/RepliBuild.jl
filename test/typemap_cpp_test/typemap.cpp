#include "typemap.hpp"

namespace tmap {

static Color store[2] = { Color::Green, Color::Blue };

void color_out(Color &c) { c = Color::Blue; }
bool color_is(const Color &c, Color d) { return c == d; }
void level_out(Level *out) noexcept { *out = Level::High; }
int  levels_sum(const Level *xs, int n) { int s = 0; for (int i = 0; i < n; i++) s += static_cast<int>(xs[i]); return s; }
void color_swap(Color &a, Color &b) noexcept { Color t = a; a = b; b = t; }
Color *color_store() { return store; }
Color &color_first() { return store[0]; }
Palette palette_make() { return Palette{ store, 2, Level::Low }; }

char8_t  u8_echo(char8_t c) { return c; }
char16_t c16_echo(char16_t c) { return c; }
char32_t c32_echo(char32_t c) { return c; }
char16_t c16_next(char16_t c) noexcept { return static_cast<char16_t>(c + 1); }
long chars_mix(char8_t a, char16_t b, char32_t c, int tail) {
    return static_cast<long>(a) + 1000L * static_cast<long>(b) + 100000000L * static_cast<long>(c % 7) + tail;
}

std::byte byte_flip(std::byte b) { return ~b; }
int  bytes_sum(const std::byte *p, int n) { int s = 0; for (int i = 0; i < n; i++) s += std::to_integer<int>(p[i]); return s; }
void bytes_fill(std::byte *p, int n) noexcept { for (int i = 0; i < n; i++) p[i] = static_cast<std::byte>(i + 1); }
Handle handle_next(Handle h) { return static_cast<Handle>(static_cast<std::uint32_t>(h) + 1); }
void opaque_set(Opaque *o) noexcept { *reinterpret_cast<int *>(o) = 7; }

unsigned __int128 u128_echo(unsigned __int128 v) noexcept { return v; }
unsigned __int128 u128_after5(long a, long b, long c, long d, long e, unsigned __int128 v) noexcept {
    return v + static_cast<unsigned __int128>(a + b + c + d + e);
}
unsigned __int128 u128_after7(long a, long b, long c, long d, long e, long f, long g,
                              unsigned __int128 v) noexcept {
    return v + static_cast<unsigned __int128>(a + b + c + d + e + f + g);
}
unsigned long long u128_hi(unsigned __int128 v) noexcept { return static_cast<unsigned long long>(v >> 64); }

} // namespace tmap
