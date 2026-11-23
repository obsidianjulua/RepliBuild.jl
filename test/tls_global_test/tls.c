/* Thread-local globals, C (test_tls_globals_live.jl). Each thread has its
 * own instance of a `_Thread_local` or `__thread` variable; the functions
 * report the calling thread's. `plain_count` is an ordinary global, one
 * instance for the process. */

_Thread_local int tls_counter = 7;
__thread double tls_scale = 1.5;
int plain_count = 3;

int *tls_counter_addr(void) { return &tls_counter; }
int tls_counter_get(void) { return tls_counter; }
void tls_counter_set(int v) { tls_counter = v; }
double *tls_scale_addr(void) { return &tls_scale; }
int *plain_count_addr(void) { return &plain_count; }
