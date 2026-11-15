# Subprocess probe for test_inline_ctors_live.jl: builds each class whose
# constructor the build's forced unit defined, on caller storage sized by
# `struct_size`, and reads a field back through the library. In a child so a
# wrong constructor binding cannot take the suite down. Prints one
# "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_inline_ctor.jl <wrapper_path>

wrapper, = ARGS
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

# Storage for one object of the class, as a caller would allocate it.
function with_storage(f, sym::Symbol)
    n = M.struct_size(sym)
    mem = Libc.malloc(n)
    try
        f(mem)
    finally
        Libc.free(mem)
    end
end

# A header-only constructor the .cpp never used: forced, bound, and it builds
# the object the library reads. Two of them on one class.
probe("plain_forced_constructor") do p
    with_storage(:Plain) do mem
        M.ic_Plain_Plain(mem, 21)
        (v = M.ic_plain_value(mem)) == 21 || push!(p, "plain_value after Plain(21) = $v")
        M.ic_Plain_Plain(mem, 6, 7)
        (w = M.ic_plain_value(mem)) == 42 || push!(p, "plain_value after Plain(6, 7) = $w")
    end
end

# A default argument is the caller's to pass: the symbol spells both parameters.
probe("default_argument_spelled") do p
    with_storage(:Dflt) do mem
        M.ic_Dflt_Dflt(mem, 4, 3)
        (s = M.ic_dflt_sum(mem)) == 7 || push!(p, "dflt_sum = $s")
    end
end

# An explicit constructor: direct-initialization through placement new.
probe("explicit_constructor") do p
    with_storage(:Expl) do mem
        M.ic_Expl_Expl(mem, 1.5f0)
        (f = M.ic_expl_f(mem)) == 1.5f0 || push!(p, "expl_f = $f")
    end
end

# A constructor the library defined itself is bound as before.
probe("defined_constructor_untouched") do p
    with_storage(:Defined) do mem
        M.ic_Defined_Defined(mem, 5)
        (v = M.ic_defined_value(mem)) == 10 || push!(p, "defined_value = $v")
    end
    with_storage(:Del) do mem
        M.ic_Del_Del(mem, 9)
        (v = M.ic_del_value(mem)) == 9 || push!(p, "del_value = $v")
    end
end

# What the unit must NOT have produced a binding for: the abstract class's
# constructor (the guard declined it), the private one, the deleted one.
probe("declined_and_skipped_unbound") do p
    for name in (:ic_Abs_Abs, :ic_Priv_Priv, :ic_Del_Del)
        if name == :ic_Del_Del
            # Del(int) is the library's own; Del() must have no zero-argument method.
            any(m -> length(m.sig.parameters) == 2, methods(M.ic_Del_Del)) &&
                push!(p, "ic_Del_Del has a method for the deleted Del()")
        else
            isdefined(M, name) && push!(p, "$name is bound")
        end
    end
end

println("PROBE_DONE")
