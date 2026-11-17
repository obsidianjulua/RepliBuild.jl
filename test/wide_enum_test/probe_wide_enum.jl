# Subprocess probe for test_wide_enums_live.jl: loads a wrapper whose enums
# have enumerators above typemax(Int64) and passes them through the library.
# Prints one "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_wide_enum.jl <wrapper_path>

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

# The enumerator that used to end the wrap: its value, through the library
# both ways, and its name when printed.
probe("value_above_int64_round_trips") do p
    UInt64(M.StatusIterationMask) == 0xffff000000000000 || push!(p, "StatusIterationMask = $(UInt64(M.StatusIterationMask))")
    M.status_bits(M.StatusIterationMask) == 0xffff000000000000 || push!(p, "status_bits = $(M.status_bits(M.StatusIterationMask))")
    M.status_mask() === M.StatusIterationMask || push!(p, "status_mask() = $(M.status_mask())")
    string(M.status_mask()) == "StatusIterationMask" || push!(p, "prints as $(string(M.status_mask()))")
    M.status_bits(M.StatusSplitShift) == 32 || push!(p, "StatusSplitShift = $(M.status_bits(M.StatusSplitShift))")
    # An open enum takes any value of its underlying type.
    M.status_bits(M.Status(typemax(UInt64))) == typemax(UInt64) || push!(p, "typemax(UInt64) did not pass")
end

# A flag set whose top bit is bit 63 is an integer alias, as flag sets are.
probe("flag_set_through_bit_63") do p
    M.Wide === UInt64 || push!(p, "Wide is $(M.Wide)")
    M.WideTop == UInt64(1) << 63 || push!(p, "WideTop = $(M.WideTop)")
    M.wide_top_set(M.WideTop) == 1 || push!(p, "wide_top_set(WideTop) = $(M.wide_top_set(M.WideTop))")
    M.wide_top_set(M.WideA | M.WideB) == 0 || push!(p, "wide_top_set(A|B) set")
    M.wide_all() == (M.WideA | M.WideB | M.WideC | M.WideTop) || push!(p, "wide_all() = $(M.wide_all())")
end

# An ordinary flag set is what it was.
probe("small_flag_set_unchanged") do p
    M.Small === Cuint || push!(p, "Small is $(M.Small)")
    M.small_bits(M.SmallA | M.SmallC) == 5 || push!(p, "small_bits = $(M.small_bits(M.SmallA | M.SmallC))")
end

println("PROBE_DONE")
