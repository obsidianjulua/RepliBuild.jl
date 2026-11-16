# Subprocess probe for test_visible_records_live.jl: builds each class
# `[types] records` made visible, through the constructor the build's forced
# unit then defined, and reads it back through the library. In a child so a
# wrong binding cannot take the suite down. Prints one
# "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_visible_record.jl <wrapper_path>

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

function with_storage(f, sym::Symbol)
    mem = Libc.malloc(M.struct_size(sym))
    try
        f(mem)
    finally
        Libc.free(mem)
    end
end

# The class has a type with its layout, and its header-only constructor builds it.
probe("listed_record_constructs") do p
    M.struct_size(:Unused) == 4 || push!(p, "struct_size(:Unused) = $(M.struct_size(:Unused))")
    fieldnames(M.Unused) == (:v,) || push!(p, "Unused fields: $(fieldnames(M.Unused))")
    with_storage(:Unused) do mem
        M.vr_Unused_Unused(mem, 21)
        (v = M.vr_unused_value(mem)) == 42 || push!(p, "unused_value after Unused(21) = $v")
    end
end

# A nested class, listed by its qualified spelling.
probe("nested_record_constructs") do p
    with_storage(:Inner) do mem
        M.vr_Outer_Inner_Inner(mem, 1.5f0)
        (f = M.vr_inner_value(mem)) == 1.5f0 || push!(p, "inner_value = $f")
    end
end

# What the library used all along is bound as before.
probe("used_record_untouched") do p
    with_storage(:Used) do mem
        M.vr_Used_Used(mem, 7)
        (v = M.vr_used_value(mem)) == 7 || push!(p, "used_value = $v")
    end
end

println("PROBE_DONE")
