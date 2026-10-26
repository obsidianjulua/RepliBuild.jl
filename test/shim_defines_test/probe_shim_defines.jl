# Subprocess probe for test_shim_defines_live.jl: calls the functions of
# stbish.h's implementation section, which only the shim file compiled, and in
# the `user` build also the library source that links against them and the two
# [wrap.macros] shims. Prints one "PROBE <name>: PASS|FAIL …" line each, then
# PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_shim_defines.jl <wrapper_path> alone|user

wrapper, mode = ARGS[1], ARGS[2]
M = include(wrapper)

function probe(f, name)
    verdict = try
        ok, got = f()
        ok ? "PASS" : "FAIL got $got"
    catch e
        "FAIL " * first(sprint(showerror, e), 300)
    end
    println("PROBE $name: $verdict")
end

probe("add") do; r = M.stbish_add(Cint(2), Cint(3)); (r == 5, r); end
probe("range") do; r = M.stbish_range(Cint(3), Cint(10)); (r.lo == 3 && r.hi == 10, r); end
probe("span") do; r = M.stbish_span(M.StbishRange(Cint(3), Cint(10))); (r == 7, r); end
if mode == "user"
    probe("user_total") do; r = M.user_total(); (r == 42, r); end
    probe("version") do; r = M.STBISH_VERSION(); (r == 7, r); end
    probe("twice") do; r = M.stbish_twice(Cint(21)); (r == 42, r); end
end
println("PROBE_DONE")
