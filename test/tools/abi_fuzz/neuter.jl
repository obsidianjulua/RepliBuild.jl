# test/tools/abi_fuzz/neuter.jl — put a known layout bug back, in-process only.
#
# The negative check for test/tools/abi_fuzz.jl: a tool that cannot see a bug
# that once shipped proves nothing by passing. Each neuter re-evaluates ONE
# method of the loaded RepliBuild with its exact signature (a less specific
# override would lose dispatch and prove nothing), replacing the fix with the
# behaviour it fixed. Nothing on disk changes: the fix is back as soon as the
# process exits.
#
# `side` is where the neutered method runs. The C generator emits struct layouts
# at wrap time, in the parent. A C++ wrapper's JIT builds its thunk IR from the
# metadata in `__init__`, so a StructGen neuter has to be applied in the child,
# before the wrapper loads.

const NEUTERS = Dict(
    "resolve_exact_layout" => (:parent,
        "GeneratorC._resolve_exact_layout without its proof that natural alignment " *
        "reproduces every DWARF offset and the DWARF size (packed / #pragma pack " *
        "structs get named fields at the wrong offsets)"),
    "apply_dwarf_layout" => (:child,
        "StructGen._apply_dwarf_layout returning the members unpadded and never " *
        "deferring to the packed body (thunks see natural LLVM layout)"),
)

function apply_neuter(RB::Module, name::AbstractString; side::Symbol)
    haskey(NEUTERS, name) || error("unknown neuter `$name`; known: $(join(sort(collect(keys(NEUTERS))), ", "))")
    NEUTERS[name][1] === side || return false
    if name == "resolve_exact_layout"
        Core.eval(RB.Wrapper, quote
            function _resolve_exact_layout(members::Vector, byte_size::Int,
                                           resolved_layouts::Dict{String,Tuple{Int,Int}},
                                           enum_layouts::Dict{String,Tuple{Int,Int}})
                byte_size <= 0 && return nothing
                isempty(members) && return nothing
                plans = Tuple{Any,String,Int,Int,Int}[]
                for m in members
                    haskey(m, "bit_size") && return nothing
                    off_raw = get(m, "offset", nothing)
                    off_raw === nothing && return nothing
                    off = _parse_int_or_hex(off_raw)
                    jt = String(get(m, "julia_type", "Any"))
                    ct = String(strip(replace(String(get(m, "c_type", "")), r"\bconst\b" => "")))
                    lay = _field_layout(jt, ct, resolved_layouts, enum_layouts)
                    lay === nothing && return nothing
                    push!(plans, (m, lay[1], lay[2], lay[3], off))
                end
                sort!(plans, by = p -> p[5])
                # NEUTERED (abi_fuzz negative check): the offset and size proof
                # is gone, so whatever natural alignment gives is emitted.
                maxal = maximum(p[4] for p in plans)
                members′ = Vector{Any}(undef, length(plans))
                for (i, (m, ftype, fsize, _, _)) in enumerate(plans)
                    m2 = copy(m)
                    m2["julia_type"] = ftype
                    m2["size"] = fsize
                    members′[i] = m2
                end
                return (members′, maxal)
            end
        end)
    elseif name == "apply_dwarf_layout"
        Core.eval(RB.JLCSIRGenerator.StructGen, quote
            function _apply_dwarf_layout(name::String, info::Any, emit::Vector{String},
                                         measure::Vector{String}, all_structs,
                                         dwarf_size::Int, is_packed::Bool)
                # NEUTERED (abi_fuzz negative check): no DWARF padding, and
                # never `nothing`, so packed layouts are never taken.
                return emit
            end
        end)
    end
    @warn "abi_fuzz: neuter `$name` applied ($side): $(NEUTERS[name][2])"
    return true
end
