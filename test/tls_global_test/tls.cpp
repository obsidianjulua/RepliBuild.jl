// Thread-local globals, C++ (test_tls_globals_live.jl). `cfg::depth` is a
// namespace member with a constant initializer; `dyn` is initialized by a
// call, which C++ runs in each thread the first time the variable is used
// there (through the variable's thread-local wrapper function); `Slot::n` is
// a static member, which a unit cannot declare outside its class.

static int init_calls = 0;
static int next_dyn() noexcept { return 40 + ++init_calls; }

namespace cfg { thread_local int depth = 4; }
thread_local int dyn = next_dyn();
struct Slot { static thread_local int n; int pad; };
thread_local int Slot::n = 9;

int *depth_addr() noexcept { return &cfg::depth; }
int *dyn_addr() noexcept { return &dyn; }
int *slot_n_addr() noexcept { return &Slot::n; }
int dyn_init_calls() noexcept { return init_calls; }
