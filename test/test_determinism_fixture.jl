# =============================================================================
# Emission determinism on a real build: C and C++ units, libc++
#
# test_emission_determinism.jl (runtests) drives each case from synthetic DWARF.
# This builds the shapes the llama.cpp dry run measured, from source, and checks
# them on what clang and GNU binutils actually print:
#
#   * dt_traits and dt_block are defined by a C unit and a C++ unit, which spell
#     them differently (`_Bool`/`bool`, anonymous `_anon1`/named `data`);
#   * dt_model::impl and dt_vocab::impl share the bare name `impl`;
#   * dt_hooks_set stores a lambda in a std::function, so libc++'s
#     `__func<(lambda at <path>:<line>:<col>), …>` names the closure type, and
#     dt_hooks itself has an anonymous union member;
#   * dt_shape's vtable home emits `__clang_vtable`;
#   * dt_run_with takes a callback declared with a typedef.
#
# The extracted tables must be the same at every DIE offset and in every unit
# order of the same DWARF. The closure must be gone and dt_hooks' union kept,
# the shared `impl` marked, the IR files named by SHA-256. The wrapper's Managed
# blocks must be sorted, its callback example must be the typedef's, and its text
# must not change when the metadata file lists its keys in another order.
#
# Needs clang++ (the C++ units) and libJLCS (the C++ wrapper's Tier 2 path).
# =============================================================================

using Test
using RepliBuild
import JSON
import SHA

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const DT_DIR = joinpath(@__DIR__, "determinism_test")
const DT_C = RepliBuild.Compiler

# The dump's units: the text before the first one, then each unit's start offset
# and its lines.
function _dt_units(dump::AbstractString)
    head = String[]
    units = Tuple{Int,Vector{String}}[]
    for l in split(dump, '\n')
        m = match(r"^\s*Compilation Unit @ offset (0x[0-9a-f]+|\d+):", l)
        if m !== nothing
            push!(units, (parse(Int, m.captures[1]), String[]))
        end
        push!(isempty(units) ? head : last(units[end]), String(l))
    end
    return head, units
end

# Every DIE header offset and every `<0x…>` reference moved by `newoff`.
function _dt_rewrite(lines, newoff)
    hex(v) = string(v; base = 16)
    out = IOBuffer()
    for l in lines
        l = replace(l, r"^(\s*<\d+><)([0-9a-f]+)>" => s -> begin
            m = match(r"^(\s*<\d+><)([0-9a-f]+)>", s)
            string(m.captures[1], hex(newoff(parse(Int, m.captures[2]; base = 16))), ">")
        end)
        l = replace(l, r"<0x([0-9a-f]+)>" => s -> "<0x" * hex(newoff(parse(Int, s[4:end-1]; base = 16))) * ">")
        println(out, l)
    end
    return String(take!(out))
end

# The same DWARF with its units in `order` (a permutation of 1:n) and every
# offset moved to where that layout puts it: a link order change. `shift` moves
# everything further, as an upstream edit does.
function _dt_relayout(dump, tables, order; shift::Integer = 0)
    head, units = _dt_units(dump)
    starts = first.(units)
    ends = [i < length(units) ? starts[i + 1] : starts[i] + 0x100000 for i in eachindex(units)]
    newstart = Dict{Int,Int}()
    at = starts[1] + Int(shift)
    for i in order
        newstart[i] = at
        at += ends[i] - starts[i]
    end
    unit_of(off) = something(findlast(<=(off), starts), 1)
    newoff(off) = off - starts[unit_of(off)] + newstart[unit_of(off)]
    body = IOBuffer()
    print(body, join(head, '\n'), '\n')
    for i in order
        lines = copy(last(units[i]))
        lines[1] = replace(lines[1], r"offset (0x[0-9a-f]+|\d+):" => "offset 0x$(string(newstart[i]; base = 16)):")
        print(body, _dt_rewrite(lines, newoff))
    end
    return String(take!(body)), tables[order]
end

_dt_tables(dump, tables) = DT_C.parse_dwarf_dump(dump; file_tables = tables)

# A JSON file whose objects list their keys in reverse: what a metadata file
# written by another Julia release (another Dict order) looks like.
function _dt_write_reversed(io, x::AbstractDict)
    print(io, "{")
    for (i, k) in enumerate(reverse(sort!(collect(keys(x)); by = string)))
        i > 1 && print(io, ",")
        print(io, JSON.json(string(k)), ":")
        _dt_write_reversed(io, x[k])
    end
    print(io, "}")
end
function _dt_write_reversed(io, x::AbstractVector)
    print(io, "[")
    for (i, v) in enumerate(x)
        i > 1 && print(io, ",")
        _dt_write_reversed(io, v)
    end
    print(io, "]")
end
_dt_write_reversed(io, x) = print(io, JSON.json(x))

_dt_strip_times(code) = join(filter(l -> !occursin(r"Generated: |generated_at", l), split(code, '\n')), '\n')

if requires("emission determinism fixture", :libJLCS, :clangxx)
@testset "Emission determinism on a real C/C++ build" begin
    toml_path = joinpath(DT_DIR, "replibuild.toml")
    srcs = [joinpath(DT_DIR, "src", f) for f in ("dt_c_part.c", "dt_model.cpp", "dt_vocab.cpp")]
    inc = joinpath(DT_DIR, "include")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "determinism"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(DT_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC"]
    source_files = [$(join(("\"$(escape_string(s))\"" for s in srcs), ", "))]
    include_dirs = ["$(escape_string(inc))"]

    [link]
    enable_lto = false
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    language = "cpp"

    [types]
    strictness = "warn"
    allow_unknown_structs = true
    allow_function_pointers = true

    [cache]
    enabled = false
    """)

    cfg = RepliBuild.ConfigurationManager.load_config(toml_path)
    build_dir = RepliBuild.ConfigurationManager.get_build_path(cfg)
    # From nothing: IR files left by an earlier run would answer for this one.
    rm(build_dir; recursive = true, force = true)
    RepliBuild.build(toml_path)
    lib = joinpath(RepliBuild.ConfigurationManager.get_output_path(cfg),
                   RepliBuild.ConfigurationManager.get_library_name(cfg))
    @test isfile(lib)
    meta_path = joinpath(dirname(lib), "compilation_metadata.json")
    meta = RepliBuild._read_json(meta_path)
    sd = meta["struct_definitions"]

    @testset "IR files are named by SHA-256 of their directory" begin
        # The digest computed here, apart from `_ir_dir_key`: SHA-256 over
        # length-prefixed fields (`_key_feed!`'s format), the same on every Julia.
        function sha_key(dir)
            ctx = SHA.SHA256_CTX()
            for field in ("replibuild-ir-dir-v1", DT_C._canon_path(dir))
                bytes = codeunits(String(field))
                SHA.update!(ctx, reinterpret(UInt8, [UInt64(length(bytes))]))
                SHA.update!(ctx, bytes)
            end
            return first(bytes2hex(SHA.digest!(ctx)), 8)
        end
        for s in srcs
            want = basename(s) * "-" * sha_key(dirname(abspath(s))) * ".ll"
            have = filter(f -> startswith(f, basename(s) * "-") && endswith(f, ".ll"), readdir(build_dir))
            @test (basename(s), have) == (basename(s), [want])
        end
    end

    # The raw dump the metadata came from.
    dumper = DT_C._dwarf_dumper()
    @test dumper !== nothing
    (dump, ec) = RepliBuild.BuildBridge.execute(dumper.tool, [dumper.dwarf * "info", lib])
    @test ec == 0
    dump = replace(dump, "\r\n" => "\n")    # GNU objdump on Windows writes CRLF
    tables = DT_C._dwarf_file_tables(dumper, lib)
    _, units = _dt_units(dump)
    n = length(units)

    @testset "the DWARF has the shapes under test" begin
        # Not vacuous: one C unit and two C++ units, each defining what it should.
        @test n == 3
        @test count(l -> occursin(r"DW_AT_name\s*:.*: dt_traits$", l), split(dump, '\n')) >= 2
        @test occursin(r"DW_AT_name\s*:.*: _Bool\n", dump) && occursin(r"DW_AT_name\s*:.*: bool\n", dump)
        @test occursin("(lambda at ", dump)
        # The dynamic class. clang 23 also lists `__clang_vtable` in its vtable
        # home, a spelling `_canonical_definition` absorbs; clang 22 does not, so
        # only what every clang emits is asserted.
        @test occursin(r"DW_AT_name\s*:.*: _vptr\$dt_shape\n", dump)
        @test count(l -> occursin(r"DW_AT_name\s*:.*: impl$", l), split(dump, '\n')) >= 2
    end

    @testset "the same tables at every offset and in every unit order" begin
        ref = _dt_tables(dump, tables)
        reftext = DT_C._canonical_text(Any[ref...])
        # Everything, when only the offsets move.
        for shift in (0x40, 0x1230, 0x7ffc0)
            d2, t2 = _dt_relayout(dump, tables, collect(1:n); shift)
            @test DT_C._canonical_text(Any[_dt_tables(d2, t2)...]) == reftext
        end
        # The type tables, when the units are reordered too: struct and enum
        # definitions, globals, typedefs. (Which of a function's declaration and
        # definition DIEs is read first does follow the unit order.)
        types(t) = DT_C._canonical_text(Any[t[2], t[3], t[4]])
        reftypes = types(ref)
        orders = [[3, 2, 1], [2, 1, 3], [2, 3, 1]]
        for o in orders
            d2, t2 = _dt_relayout(dump, tables, o; shift = 0x1000)
            @test (o, types(_dt_tables(d2, t2)) == reftypes) == (o, true)
        end
    end

    @testset "metadata: one definition per name, chosen by content" begin
        # The C unit's spellings sort first.
        @test [m["c_type"] for m in sd["dt_traits"]["members"]] == ["int", "_Bool"]
        @test [m["name"] for m in sd["dt_block"]["members"]] == ["_anon1", "qs"]
        @test !haskey(sd["dt_traits"], "conflict")
        @test !haskey(sd["dt_block"], "conflict")
        @test haskey(sd, "dt_block_anon1") && haskey(sd, "dt_block_data")
        # dt_model::impl and dt_vocab::impl.
        @test get(sd["impl"], "conflict", false) == true
    end

    @testset "metadata: the closure is gone, the project's anonymous union is not" begin
        @test !any(k -> occursin("lambda", k), keys(sd))
        @test haskey(sd, "dt_hooks")
        @test haskey(sd, "dt_hooks_u")
        @test get(sd["dt_hooks_u"], "anonymous", false) == true
    end

    @testset "metadata: the callback parameter names its typedef" begin
        f = only(filter(f -> f["name"] == "dt_run_with", meta["functions"]))
        cb = only(filter(p -> p["name"] == "cb", f["parameters"]))
        @test cb["function_pointer_typedef"] == "dt_progress_cb"
    end

    RepliBuild.wrap(toml_path)
    wrapper = joinpath(DT_DIR, "julia", "Determinism.jl")
    @test isfile(wrapper)
    code = read(wrapper, String)

    @testset "wrapper: sorted Managed blocks and globals, no lambda" begin
        managed = [m.captures[1] for m in eachmatch(r"\nmutable struct Managed(\w+)\n", code)]
        @test length(managed) >= 5
        @test issorted(managed)
        gpos = [something(findfirst("function $g()", code), 0:0)[1] for g in ("dt_g_alpha", "dt_g_mid", "dt_g_zeta")]
        @test all(>(0), gpos)
        @test issorted(gpos)
        @test !occursin("lambda", code)
        @test occursin("struct dt_hooks_u", code)
    end

    @testset "wrapper: a callback's example is its typedef's, or there is none" begin
        @test occursin("@cfunction(my_callback, Cint, (Cint, Ptr{Cvoid},))", code)
        # `dt_run`, the decoy the old scoring preferred for both functions.
        @test !occursin("Cdouble, (Cdouble,)", code)
        # dt_run_raw's `cb` has no typedef: one example in the file, dt_run_with's.
        @test count("**Callback `cb`**", code) == 1
    end

    @testset "wrapper: the same text when the metadata's keys come in another order" begin
        open(meta_path, "w") do io
            _dt_write_reversed(io, meta)
        end
        RepliBuild.wrap(toml_path)
        @test _dt_strip_times(read(wrapper, String)) == _dt_strip_times(code)
    end
end
end # requires
