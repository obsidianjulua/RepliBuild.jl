# Subprocess probe for test_nested_type_aliases_live.jl: uses class-nested
# types under the names their functions carry. In a child so a wrong binding
# cannot take the suite down. Prints one "PROBE <case>: PASS|FAIL …" line
# each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_nested_type.jl <wrapper_path>

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

# The alias IS the type, and the bare name is still bound.
probe("alias_is_the_type") do p
    M.World_RayResult === M.RayResult || push!(p, "World_RayResult is not RayResult")
    M.World_Callback_Info === M.Info || push!(p, "World_Callback_Info is not Info")
    M.nt_Box_Lid === M.Lid || push!(p, "nt_Box_Lid is not Lid")
    fieldnames(M.World_RayResult) == (:fraction, :id) || push!(p, "fields: $(fieldnames(M.World_RayResult))")
end

# struct_size and member_offset answer to both names.
probe("layout_by_both_names") do p
    M.struct_size(:World_RayResult) == M.struct_size(:RayResult) == 8 ||
        push!(p, "struct_size: $(M.struct_size(:World_RayResult)) vs $(M.struct_size(:RayResult))")
    M.member_offset(:World_RayResult, :id) == M.member_offset(:RayResult, :id) == 4 ||
        push!(p, "member_offset(:World_RayResult, :id) = $(M.member_offset(:World_RayResult, :id))")
    M.struct_size(:nt_Box_Lid) == 4 || push!(p, "struct_size(:nt_Box_Lid) = $(M.struct_size(:nt_Box_Lid))")
end

# One name through a whole call: storage by the qualified name, the
# constructor and the method that carry it, and the value read back as it.
probe("one_name_end_to_end") do p
    mem = Libc.malloc(M.struct_size(:World_RayResult))
    try
        M.World_RayResult_RayResult(mem, 0.25f0, 7)
        M.World_RayResult_hit(mem) == true || push!(p, "hit() false for fraction 0.25")
        r = unsafe_load(Ptr{M.World_RayResult}(mem))
        (r.fraction, r.id) == (0.25f0, 7) || push!(p, "read back $r")
        w = Ref(M.World(10))
        (c = M.World_cast(w, Ref(r))) == 17 || push!(p, "World::cast = $c")
    finally
        Libc.free(mem)
    end
end

# A name the module already binds is not rebound: `World_Mark` is the free
# function, and `Mark` keeps its bare name alone.
probe("taken_name_not_rebound") do p
    M.World_Mark isa Function || push!(p, "World_Mark is $(typeof(M.World_Mark))")
    haskey(M.STRUCT_SIZES, :World_Mark) && push!(p, "STRUCT_SIZES has :World_Mark")
    mem = Libc.malloc(M.struct_size(:Mark))
    try
        M.World_Mark_Mark(mem, 5)
        (v = M.World_Mark(mem)) == 5 || push!(p, "World_Mark(mark) = $v")
    finally
        Libc.free(mem)
    end
    # A namespace is not a class: no alias for nt::Plain or nt::Box.
    for n in (:nt_Plain, :nt_Box)
        isdefined(M, n) && push!(p, "$n is bound")
    end
end

println("PROBE_DONE")
