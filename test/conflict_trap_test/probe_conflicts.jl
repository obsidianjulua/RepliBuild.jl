# Subprocess probe for test_conflict_trap_fixture.jl: calls every function of
# conflict_trap_test (`cpp`) or conflict_trap_c_test (`c`). A function that a
# name several types share crosses by value must raise the ABI Safety Trap; every
# other one must return the right value. Prints one "PROBE <function>: PASS|FAIL …"
# line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_conflicts.jl <wrapper_path> <cpp|c>

wrapper, lang = ARGS
const M = include(wrapper)

function probe(f, name)
    verdict = try
        p = String[]
        f(p)
        isempty(p) ? "PASS" : "FAIL " * join(p, "; ")
    catch e
        "FAIL threw " * first(sprint(showerror, e), 300)
    end
    println("PROBE $name: ", verdict)
end

# Some value of each declared argument type: a trap raises before reading it.
zero_of(T) = (T === Any || !isconcretetype(T)) ? 0 : reinterpret(T, zeros(UInt8, sizeof(T)))[1]

function expect_trap(p, fname::Symbol)
    f = getfield(M, fname)
    args = map(zero_of, collect(only(methods(f)).sig.parameters)[2:end])
    try
        r = f(args...)
        push!(p, "$fname returned $r")
    catch e
        msg = sprint(showerror, e)
        e isa ErrorException || push!(p, "$fname threw $(typeof(e)): $(first(msg, 200))")
        occursin("ABI Safety Trap: cannot call '$fname'", msg) || push!(p, "$fname: $(first(msg, 200))")
        occursin("bare DWARF name", msg) || push!(p, "$fname: the message does not say why")
    end
end

if lang == "cpp"
    probe("ct_duo_sum") do p
        (r = M.ct_duo_sum(M.Duo(1, 2))) == 5 || push!(p, "ct_duo_sum = $r")
    end
    # `Mode` is beta's or alpha's enum in Julia; both are 4 bytes and both have a 1.
    probe("ct_mode_value") do p
        (r = M.ct_mode_value(M.Mode(1))) == 101 || push!(p, "ct_mode_value = $r")
    end
    probe("ct_impl_ptr_sum") do p
        buf = Int32[3, 4]                                  # alpha::impl {3, 4}
        r = GC.@preserve buf M.ct_impl_ptr_sum(Ptr{M.impl}(pointer(buf)))
        r == 34 || push!(p, "ct_impl_ptr_sum = $r")
    end
    probe("ct_duo_twice") do p
        (r = M.ct_duo_twice(M.Duo(1, 2))) == 6 || push!(p, "ct_duo_twice = $r")
    end
    probe("ct_beta_mode") do p
        (r = M.ct_beta_mode(M.Mode(1))) == 8 || push!(p, "ct_beta_mode = $r")
    end
    for f in (:ct_impl_sum, :ct_holder_sum, :ct_alpha_kind, :ct_vocab_score, :ct_kind)
        probe(p -> expect_trap(p, f), String(f))
    end
else
    probe("pt_sum") do p
        (r = M.pt_sum(M.pt(1, 2))) == 5 || push!(p, "pt_sum = $r")
    end
    for f in (:st_one, :st_two)
        probe(p -> expect_trap(p, f), String(f))
    end
end
println("PROBE_DONE")
