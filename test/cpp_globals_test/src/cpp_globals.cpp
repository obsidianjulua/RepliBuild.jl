#include "cpp_globals.h"

int g_top = 9;
namespace a { int x = 7; }
namespace b { int x = 8; }
namespace cfg { int depth = 4; const double scale = 1.25; }
std::string g_string("global-string");

int Stat::count = 0;
const double Stat::factor = 2.5;
int Stat::prot_ = 3;
int Stat::bump() { return ++count + prot_ - 3; }

namespace geo {
int Vec3::live_count = 0;
Vec3::Vec3() : x(0), y(0), z(0) { ++live_count; }
Vec3::~Vec3() { --live_count; }
int Counter::total_ = 11;
int Counter::total() { return total_; }
Vec3 *vec3_new() { return new Vec3(); }
void vec3_free(Vec3 *v) { delete v; }
}
