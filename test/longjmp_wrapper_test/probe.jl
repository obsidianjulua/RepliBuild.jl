# probe.jl <wrapper> <rounds>: a Julia callback leaves through a wrapped C
# function that longjmps, and the collector runs while the frame that made the
# protected call is still live. A wrapper that held a root across the jump
# leaves the task's GC stack pointing into abandoned stack memory; this probe
# overwrites that memory and collects, so such a wrapper is a crash here.
#
# One `PROBE <name>: PASS|FAIL …` line per case, then `PROBE_DONE`.
const M = Base.include(Module(), ARGS[1])
const ROUNDS = parse(Int, ARGS[2])
probe(name, ok, detail = "") = (println("PROBE ", name, ": ", ok ? "PASS" : "FAIL " * string(detail)); flush(stdout))

cb_plain(::Ptr{Cvoid})::Cint = (M.lj_fail("left by longjmp"); Cint(0))
cb_format(::Ptr{Cvoid})::Cint = (M.lj_failf_Cint("left with a code", 42); Cint(0))
cb_returns(::Ptr{Cvoid})::Cint = Cint(7)

# Fills stack below the caller with a pattern, where the abandoned frames were.
@noinline function scribble(depth::Int, seed::UInt)
    t = ntuple(i -> seed + UInt(i), Val(64))
    depth == 0 && return sum(t)
    return scribble(depth - 1, seed ⊻ t[17]) + t[3]
end

# Everything in one frame, as a test file's top-level block is one frame.
function rounds(cf, n)
    landed = 0
    for i in 1:n
        landed += M.lj_protected(cf, C_NULL)
        scribble(24, UInt(0xdeadbeef00000000) + UInt(i))
        junk = [string(i, '-', j) for j in 1:200]
        i % 10 == 0 && GC.gc()
        landed += length(junk) == 200 ? 0 : 1000
    end
    return landed
end

function main()
    cf = @cfunction(cb_plain, Cint, (Ptr{Cvoid},))
    n = rounds(cf, ROUNDS)
    probe("plain", n == ROUNDS && M.lj_last_message() == "left by longjmp", (n, M.lj_last_message()))

    ff = @cfunction(cb_format, Cint, (Ptr{Cvoid},))
    n = rounds(ff, ROUNDS)
    probe("variadic", n == ROUNDS && M.lj_last_message() == "left with a code" && M.lj_last_code() == 42,
          (n, M.lj_last_message(), M.lj_last_code()))

    rf = @cfunction(cb_returns, Cint, (Ptr{Cvoid},))
    probe("returns", rounds(rf, ROUNDS) == 0)
    GC.gc(); GC.gc()
    probe("still_calls", M.lj_len("after the jumps") == 15)
end
main()
println("PROBE_DONE")
