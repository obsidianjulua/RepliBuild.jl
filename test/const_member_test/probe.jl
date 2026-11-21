# probe.jl <wrapper>: call the const-member fixture through its wrapper, in a
# process of its own (before the fix the library's JIT did not initialise;
# a wrong by-value crossing is a crash or a wrong number).
#
# One `PROBE <name>: PASS|FAIL …` line per case, then `PROBE_DONE`.
const M = Base.include(Module(), ARGS[1])
probe(name, ok, detail = "") = (println("PROBE ", name, ": ", ok ? "PASS" : "FAIL " * string(detail)); flush(stdout))
attempt(f, name) = try
    f()
catch e
    probe(name, false, first(split(sprint(showerror, e), '\n')))
end

function main()
    # The first Tier 2 call creates the library's engine from its whole thunk
    # module: one thunk that does not lower fails every one of them.
    attempt("module") do
        probe("module", M.checked(41) == 42)
    end

    sfx = "sfx"
    namer = Ref(M.Namer(M.KIND_C))
    GC.@preserve sfx namer begin
        attempt("name_returned") do
            n = M.Namer_make(namer, M.KIND_B, pointer(sfx), 3, 4)
            got = (n.arch, n.tensor, unsafe_string(n.suffix), n.bid, n.xid)
            probe("name_returned", got == (M.KIND_C, M.KIND_B, "sfx", 3, 4), got)
        end
        attempt("name_operator") do
            n = M.Namer_operatorcall(namer, M.KIND_A, 7, 9)
            got = (n.arch, n.tensor, n.suffix == C_NULL, n.bid, n.xid)
            probe("name_operator", got == (M.KIND_C, M.KIND_A, true, 7, 9), got)
        end
        attempt("name_passed") do
            n = M.Name(M.KIND_C, M.KIND_B, pointer(sfx), Cint(3), Cint(4))
            # 7·10⁶ + 2·10⁴ + 3·10² + 4, and 3·10⁸ for the suffix's length.
            got = M.name_total(n)
            back = M.name_total(M.Namer_make(namer, M.KIND_B, pointer(sfx), 3, 4))
            probe("name_passed", (got, back) == (307020304, 307020304), (got, back))
        end
    end

    attempt("pair") do
        P = getfield(M, :c_Pair)        # `Pair` is Base's: the wrapper names it c_Pair
        p = M.pair_of(-5, 7)
        got = (p.a, p.b, M.pair_sum(P(Cint(-5), Cuint(7))), M.pair_sum(p))
        probe("pair", got == (-5, 7, 2, 2), got)
    end

    attempt("mixed") do
        m = M.mixed_of(3, 10, 2.5)
        got = (m.tag, m.count, m.scale, M.mixed_total(m))
        probe("mixed", got == (3, 10, 2.5, 15.5), got)
    end

    # A record whose body is a byte region: opaque here, so it is checked by
    # sending back what the library returned. Where the target cannot class
    # the record by value at all (AAPCS64 has to rule out a homogeneous float
    # aggregate at any size, and one member's kind is unknown), the wrapper
    # refuses the call, and that refusal is the right answer: `PASS refused`.
    attempt("byte_body") do
        got = try
            (M.wide_sum(M.wide_of(5, 9, 1000, -4)), M.wide_sum(M.wide_of(0xabcdef, -1, 1 << 40, 3)))
        catch e
            e
        end
        if got isa Exception
            msg = sprint(showerror, got)
            occursin("ABI Safety Trap", msg) ? println("PROBE byte_body: PASS refused") :
                                               probe("byte_body", false, first(split(msg, '\n')))
            flush(stdout)
        else
            probe("byte_body", got == (1010, 0xabcdef - 1 + (1 << 40) + 3), got)
        end
    end

    attempt("qualified_scalars") do
        got = (M.scaled(2.5, 4), M.halved(3.0f0), M.doubled(1.25), M.mixed_args(1, 20.0, 300, 4000.0f0),
               M.narrow(UInt8(200), Int8(-3), -1000, true))
        probe("qualified_scalars", got == (10.0, 1.5f0, 2.5, 4321, -802), got)
    end

    attempt("throws") do
        e = try
            M.checked(-1)
            nothing
        catch err
            err
        end
        probe("throws", e !== nothing && occursin("C++", sprint(showerror, e)), typeof(e))
    end
end
main()
println("PROBE_DONE")
