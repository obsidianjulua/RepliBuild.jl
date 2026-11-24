#include <chrono>
std::chrono::milliseconds ms_make(long n) { return std::chrono::milliseconds(n); }
long ms_count(std::chrono::milliseconds d) { return (long)d.count(); }
long ms_count_ref(const std::chrono::milliseconds& d) { return (long)d.count(); }
double secs_of(std::chrono::duration<double> d) { return d.count(); }
std::chrono::system_clock::time_point tp_epoch_plus(long s) { return std::chrono::system_clock::time_point(std::chrono::seconds(s)); }
long tp_seconds(std::chrono::system_clock::time_point t) { return (long)std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count(); }
