# Subprocess probe for test_tls_globals_live.jl: loads a wrapper of
# test/tls_global_test and compares each thread-local global's accessors
# with the library's own answer on a second thread. Started with two
# threads at least. Prints one "PROBE <case>: PASS|FAIL …" line each, then
# PROBE_DONE.
# Run: julia --project=<RepliBuild> --threads=2 probe.jl <wrapper> c|cpp

wrapper = ARGS[1]
lang = ARGS[2]

function probe(f, name)
    verdict = try
        p = String[]
        f(p)
        isempty(p) ? "PASS" : "FAIL " * join(unique(p)[1:min(end, 6)], "; ")
    catch e
        "FAIL threw " * first(replace(sprint(showerror, e), '\n' => ' '), 300)
    end
    println("PROBE $name: ", verdict)
end

# `f()` on a thread other than `avoid`, on one OS thread throughout: a task
# migrates only at a yield, and `f` must not yield.
function elsewhere(f, avoid::Int)
    for _ in 1:10_000
        t = Threads.@spawn begin
            tid = Threads.threadid()
            tid == avoid ? nothing : (tid, f())
        end
        r = fetch(t)
        r === nothing || return r
    end
    error("no task ran on a thread other than $avoid")
end

Threads.nthreads() >= 2 || error("the probe needs two threads, has $(Threads.nthreads())")

# A module that fails to load ends the probe here, with no PROBE_DONE.
const G = include(wrapper)

if lang == "c"
    # The main thread asks first: this is where a `cglobal` resolved once
    # would take its address from.
    const main = Threads.threadid()
    G.tls_counter_set(Cint(100))

    probe("main_thread") do p
        G.tls_counter_ptr() == G.tls_counter_addr() || push!(p, "ptr != addr")
        G.tls_counter() == 100 || push!(p, "value $(G.tls_counter())")
    end
    probe("other_thread_pointer") do p
        (tid, (ptr, addr)) = elsewhere(() -> (G.tls_counter_ptr(), G.tls_counter_addr()), main)
        ptr == addr || push!(p, "thread $tid: tls_counter_ptr() $ptr is not the library's $addr")
    end
    probe("other_thread_value") do p
        # A thread that never wrote sees the initial value, not main's 100.
        (tid, v) = elsewhere(() -> G.tls_counter(), main)
        v == 7 || push!(p, "thread $tid: tls_counter() = $v, expected 7")
    end
    probe("other_thread_write") do p
        (tid, (v, lib)) = elsewhere(main) do
            unsafe_store!(G.tls_counter_ptr(), Cint(55))
            (G.tls_counter(), G.tls_counter_get())
        end
        (v, lib) == (55, 55) || push!(p, "thread $tid: wrote 55, read ($v, $lib)")
        G.tls_counter() == 100 || push!(p, "main's instance changed to $(G.tls_counter())")
    end
    probe("double_other_thread") do p
        (tid, (ptr, addr, v)) = elsewhere(() -> (G.tls_scale_ptr(), G.tls_scale_addr(), G.tls_scale()), main)
        ptr == addr || push!(p, "thread $tid: tls_scale_ptr() is not the library's")
        v == 1.5 || push!(p, "tls_scale() = $v")
    end
    probe("plain_global_is_shared") do p
        (tid, ptr) = elsewhere(() -> G.plain_count_ptr(), main)
        ptr == G.plain_count_addr() || push!(p, "plain_count_ptr() differs on thread $tid")
        G.plain_count() == 3 || push!(p, "plain_count() = $(G.plain_count())")
    end
else
    const main = Threads.threadid()
    probe("namespace_member") do p
        G.cfg_depth_ptr() == G.depth_addr() || push!(p, "main: ptr != addr")
        (tid, (ptr, addr, v)) = elsewhere(() -> (G.cfg_depth_ptr(), G.depth_addr(), G.cfg_depth()), main)
        ptr == addr || push!(p, "thread $tid: cfg_depth_ptr() is not the library's")
        v == 4 || push!(p, "cfg_depth() = $v")
    end
    probe("dynamic_initialization") do p
        # The accessor goes through the variable's wrapper, so a thread whose
        # first use is the accessor runs the initializer: 41, 42, ... never 0.
        (tid, (v, ptr, addr)) = elsewhere(() -> (G.dyn(), G.dyn_ptr(), G.dyn_addr()), main)
        v > 40 || push!(p, "thread $tid: dyn() = $v, the initializer did not run")
        ptr == addr || push!(p, "thread $tid: dyn_ptr() is not the library's")
    end
    probe("static_member_refused") do p
        msg = try
            G.Slot_n_ptr(); ""
        catch e
            sprint(showerror, e)
        end
        occursin("thread-local", msg) || push!(p, "Slot_n_ptr() did not refuse: $(repr(first(msg, 200)))")
    end
end
println("PROBE_DONE")
