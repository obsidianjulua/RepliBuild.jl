# Subprocess probe for test_memberless_live.jl: crosses each memberless record
# the way a caller would, in a child so a wrong layout cannot take the suite
# down. Prints one "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_memberless.jl <wrapper_path>

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

# The Julia type IS the C layout: sizes and offsets against the DWARF tables.
probe("layout_tables_agree") do p
    for (sname, size) in M.STRUCT_SIZES
        isdefined(M, sname) || continue
        T = getfield(M, sname)
        (T isa DataType && isstructtype(T)) || continue
        sizeof(T) == size || push!(p, "$sname sizeof julia=$(sizeof(T)) dwarf=$size")
    end
    for k in (:Derived, :P, :PolyKid, :VB)
        haskey(M.STRUCT_SIZES, k) || push!(p, "no STRUCT_SIZES entry for $k")
    end
end

# A record whose bytes all come from its base crosses by value, both ways.
probe("derived_by_value") do p
    d = M.Derived((1.0f0, 2.0f0, 3.0f0, 4.0f0))
    (s = M.mb_dsum(d)) == 10.0f0 || push!(p, "dsum $s")
    r = M.mb_dmake(1.5f0)
    r.m == (1.5f0, 2.5f0, 3.5f0, 4.5f0) || push!(p, "dmake $(r.m)")
end

# An empty class is dropped by the convention: the integers after it arrive.
probe("empty_functor_dropped") do p
    (a = M.mb_pcall(M.P(), 2, 9)) == 2 || push!(p, "pcall(2, 9) = $a")
    (b = M.mb_pcall(M.P(), 9, 2)) == 2 || push!(p, "pcall(9, 2) = $b")
end

# Bullet's shape: a constructor on caller storage sized by struct_size.
probe("polykid_sized_and_constructed") do p
    n = M.struct_size(:PolyKid)
    n == 16 || push!(p, "struct_size(:PolyKid) = $n")
    mem = Libc.malloc(n)
    try
        M.mb_PolyKid_PolyKid(mem, 21)
        (v = M.mb_kid_id(mem)) == 42 || push!(p, "kid_id = $v")
    finally
        Libc.free(mem)
    end
end

# A virtual base: the entry is there (sized), the virtual base is nobody's to
# flatten, and the C++ side still works on its own objects.
probe("virtual_base_sized_not_flattened") do p
    M.struct_size(:VB) == 32 || push!(p, "struct_size(:VB) = $(M.struct_size(:VB))")
    fieldnames(M.VB)[1] == :_vptr_Poly || push!(p, "VB fields $(fieldnames(M.VB))")
    any(f -> f == :m, fieldnames(M.VB)) && push!(p, "VB flattened its virtual base")
    (v = M.mb_vb_make(3.0f0)) == 3 || push!(p, "vb_make = $v")
end

println("PROBE_DONE")
