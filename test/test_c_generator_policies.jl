# =============================================================================
# C-generator policy regressions (library-free, CI-safe)
#
# Pins the v3.0.0 audit fixes at the emission level — synthetic DWARF
# metadata drives generate_introspective_module_c directly, so no toolchain
# or library build is needed. The generated module is also loaded against
# libjulia (any real loadable library satisfies load-time isfile/dlopen) for
# pieces that don't need the fixture's symbols:
#
# 1. [wrap.cstring_owned] TOML parse (+ malformed-value rejection)
# 2. Macro shims carry __attribute__((used, visibility("default"))) — a
#    -fvisibility=hidden project must not silently drop [wrap.macros]
# 3. Cstring return policy: Union{String,Nothing}, NULL → nothing, owned
#    buffers freed via the declared symbol, raw _ptr variant exported —
#    on the base wrapper AND the vararg @ccall path
# 4. ABI trap for misaligned (packed) ≤16B all-integer blob params by value;
#    aligned blobs stay callable
# 5. Bitfield multi-byte accessors assemble the exact byte span (no
#    power-of-2 container overhanging the struct tail, no pointer stores)
# 6. Unresolved-type globals emit only a Ptr{Cvoid} _ptr accessor
# 7. Callback docs never fall back to an arbitrary typedef signature
# 8. The dead packed-struct branch stays dead (packed structs are blobs)
# =============================================================================

using Test
using RepliBuild
using Libdl

const _CM = RepliBuild.ConfigurationManager

function _policy_config(dir::String)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "policysynth"
    root = "$(dir)"

    [link]
    enable_lto = false

    [wrap]
    language = "c"
    shim_headers = ["synth.h"]

    [wrap.varargs]
    fmt_msg = [["Cint"]]

    [wrap.cstring_owned]
    make_msg = "shim_free"
    fmt_msg = "shim_free"

    [wrap.macros.SYNTH_OK]
    ret = "int"

    [wrap.macros.synth_init]
    ret = "int"
    args = ["void*", "int"]

    [types]
    strictness = "warn"
    allow_unknown_structs = true
    allow_function_pointers = true

    [cache]
    enabled = false
    """)
    return _CM.load_config(toml)
end

function _policy_metadata()
    structs = Dict{String,Any}(
        # Packed: val at offset 1 is misaligned → layout proof fails → blob;
        # all-integer, ≤16B → real SysV class is MEMORY, blob claims INTEGER
        "PackedInt" => Dict{String,Any}(
            "kind" => "struct", "byte_size" => "0x6",
            "members" => Any[
                Dict{String,Any}("name" => "tag", "julia_type" => "Cchar", "c_type" => "char", "offset" => "0x0", "size" => 1),
                Dict{String,Any}("name" => "val", "julia_type" => "Cint", "c_type" => "int", "offset" => "0x1", "size" => 4),
            ]),
        # Blob for a different reason (unresolvable member type) but aligned:
        # INTEGER class on both views → must stay callable
        "AlignedBlob" => Dict{String,Any}(
            "kind" => "struct", "byte_size" => "0x8",
            "members" => Any[
                Dict{String,Any}("name" => "m0", "julia_type" => "Any", "c_type" => "MysteryT", "offset" => "0x0", "size" => 4),
            ]),
        # 17-bit field starting at bit 45: spans bytes 5..7 of an 8-byte
        # struct — the old UInt32 container at byte 5 read/wrote byte 8 (OOB)
        "BitTail" => Dict{String,Any}(
            "kind" => "struct", "byte_size" => "0x8",
            "members" => Any[
                Dict{String,Any}("name" => "hdr", "julia_type" => "Cuint", "c_type" => "unsigned int", "offset" => "0x0", "size" => 4),
                Dict{String,Any}("name" => "x", "julia_type" => "Cuint", "c_type" => "unsigned int", "bit_size" => 17, "data_bit_offset" => 45),
            ]),
    )

    functions = Any[
        Dict{String,Any}("name" => "get_msg", "mangled" => "get_msg", "demangled" => "get_msg",
            "parameters" => Any[],
            "return_type" => Dict{String,Any}("julia_type" => "Cstring", "c_type" => "char*")),
        Dict{String,Any}("name" => "make_msg", "mangled" => "make_msg", "demangled" => "make_msg",
            "parameters" => Any[Dict{String,Any}("name" => "n", "julia_type" => "Cint", "c_type" => "int")],
            "return_type" => Dict{String,Any}("julia_type" => "Cstring", "c_type" => "char*")),
        Dict{String,Any}("name" => "fmt_msg", "mangled" => "fmt_msg", "demangled" => "fmt_msg",
            "is_vararg" => true,
            "parameters" => Any[
                Dict{String,Any}("name" => "fmt", "julia_type" => "Cstring", "c_type" => "const char*"),
                Dict{String,Any}("name" => "varargs...", "julia_type" => ""),
            ],
            "return_type" => Dict{String,Any}("julia_type" => "Cstring", "c_type" => "char*")),
        Dict{String,Any}("name" => "use_packed", "mangled" => "use_packed", "demangled" => "use_packed",
            "parameters" => Any[Dict{String,Any}("name" => "p", "julia_type" => "Any", "c_type" => "PackedInt")],
            "return_type" => Dict{String,Any}("julia_type" => "Cvoid", "c_type" => "void")),
        Dict{String,Any}("name" => "use_aligned", "mangled" => "use_aligned", "demangled" => "use_aligned",
            "parameters" => Any[Dict{String,Any}("name" => "p", "julia_type" => "Any", "c_type" => "AlignedBlob")],
            "return_type" => Dict{String,Any}("julia_type" => "Cvoid", "c_type" => "void")),
        Dict{String,Any}("name" => "on_event", "mangled" => "on_event", "demangled" => "on_event",
            "parameters" => Any[Dict{String,Any}("name" => "cb", "julia_type" => "Ptr{Cvoid}", "c_type" => "void (*)(int)")],
            "return_type" => Dict{String,Any}("julia_type" => "Cvoid", "c_type" => "void")),
    ]

    return Dict{String,Any}(
        "functions" => functions,
        "struct_definitions" => structs,
        "globals" => Dict{String,Any}(
            "g_count" => Dict{String,Any}("julia_type" => "Cint"),
            "g_mystery" => Dict{String,Any}("julia_type" => "Any"),
        ),
        # No name relates to "cb"/"on_event": the old code still documented
        # this typedef's signature via the first-entry fallback
        "function_pointer_typedefs" => Dict{String,Any}(
            "WhollyUnrelated" => Dict{String,Any}("return_type" => "int", "parameters" => Any["double"]),
        ),
    )
end

@testset "C generator policies (v3.0.0)" begin
    dir = mktempdir()
    cfg = _policy_config(dir)

    @testset "[wrap.cstring_owned] parse" begin
        @test cfg.wrap.cstring_owned == Dict("make_msg" => "shim_free", "fmt_msg" => "shim_free")

        # Non-string value → warn + ignore, never a crash or silent misparse
        baddir = mktempdir()
        badtoml = joinpath(baddir, "replibuild.toml")
        write(badtoml, """
        [project]
        name = "badowned"
        root = "$(baddir)"

        [wrap]
        language = "c"

        [wrap.cstring_owned]
        f = 3
        """)
        badcfg = @test_logs (:warn, r"cstring_owned") _CM.load_config(badtoml)
        @test isempty(badcfg.wrap.cstring_owned)
    end

    @testset "macro shims: default visibility" begin
        files = RepliBuild.Compiler.generate_macro_shims(cfg, String[])
        @test length(files) == 1
        shim_txt = read(files[1], String)
        # Both macros forced into the export table (nm -g is the wrapper's
        # symbol source; -fvisibility=hidden would otherwise drop them)
        @test occursin("__attribute__((used, visibility(\"default\"))) int replibuild_shim_SYNTH_OK()", shim_txt)
        @test occursin("__attribute__((used, visibility(\"default\"))) int replibuild_shim_synth_init(void* arg0, int arg1)", shim_txt)
        # Value macro emits the bare name; function-like macro forwards args
        @test occursin("return SYNTH_OK;", shim_txt)
        @test occursin("return synth_init(arg0, arg1);", shim_txt)
        @test occursin("#include \"synth.h\"", shim_txt)
    end

    # Any real loadable library satisfies the generated module's load-time
    # check; functions whose symbols don't exist are simply never called.
    # libjulia is loaded in every Julia process on every OS — portable anchor.
    libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    registry = RepliBuild.Wrapper.create_type_registry(cfg)
    code = RepliBuild.Wrapper.generate_introspective_module_c(
        cfg, libref, _policy_metadata(), "PolicySynth", registry, true)

    @testset "Cstring policy emission" begin
        # Base wrapper: Union return + NULL → nothing
        @test occursin("function get_msg()::Union{String,Nothing}", code)
        @test occursin("ptr == C_NULL && return nothing", code)
        # Borrowed return (not in cstring_owned): no free call in its body
        get_msg_block = match(r"function get_msg\(\)::Union\{String,Nothing\}.*?\nend"s, code)
        @test get_msg_block !== nothing
        @test !occursin("shim_free", get_msg_block.match)
        # Owned return: freed through the declared symbol after the copy
        @test occursin("ccall((:shim_free, LIBRARY_PATH), Cvoid, (Cstring,), ptr)", code)
        # Raw variants exist and are exported
        @test occursin("function get_msg_ptr()::Cstring", code)
        @test occursin(r"function make_msg_ptr\(n::Integer\)::Cstring", code)
        @test occursin(r"export [^\n]*get_msg_ptr", code)
        # Docstring shows what the wrapper returns, not the C-level type
        @test occursin("-> Union{String,Nothing}", code)

        # Vararg path: same policy on base + typed overload, @ccall form kept
        @test occursin("function fmt_msg(fmt::Cstring)::Union{String,Nothing}", code)
        @test occursin("ptr = @ccall LIBRARY_PATH.var\"fmt_msg\"(fmt::Cstring;)::Cstring", code)
        @test occursin("function fmt_msg_Cint(fmt::Cstring, va_1::Cint)::Union{String,Nothing}", code)
        @test occursin("ptr = @ccall LIBRARY_PATH.var\"fmt_msg\"(fmt::Cstring; va_1::Cint)::Cstring", code)
    end

    @testset "misaligned blob param trap" begin
        # Packed (misaligned) all-int ≤16B by value → loud trap
        @test occursin("ABI Safety Trap: cannot call 'use_packed'", code)
        @test occursin("takes `PackedInt` by value", code)
        # Aligned blob (INTEGER class both views) → normal ccall survives
        @test occursin("ccall((:use_aligned, LIBRARY_PATH)", code)
        @test !occursin("ABI Safety Trap: cannot call 'use_aligned'", code)
    end

    @testset "bitfield byte-span accessors" begin
        # Exact span: bytes 5..7 (1-based 6..8), 3 bytes, no pointer container
        @test occursin("function get_x(s::BitTail)::UInt32", code)
        @test occursin("data[6 + i]", code)
        @test occursin("for i in 0:2", code)
        @test !occursin("unsafe_store!", code)   # old setter's OOB write vector
    end

    @testset "globals + callback docs + dead branch" begin
        # Resolved global: value getter + typed pointer
        @test occursin("function g_count()::Cint", code)
        @test occursin("function g_count_ptr()::Ptr{Cint}", code)
        # Unresolved global: pointer-only, Ptr{Cvoid}, no value getter
        @test occursin("function g_mystery_ptr()::Ptr{Cvoid}", code)
        @test !occursin("function g_mystery()", code)
        # Callback with no matching typedef: honest "unknown", never the
        # arbitrary first-typedef signature
        @test occursin("signature unknown", code)
        @test !occursin("WhollyUnrelated", code)
        # Packed structs are blobs; the old dedicated packed branch is gone
        @test occursin("byte blob for ABI safety", code)
        @test !occursin("C packed struct:", code)
    end

    @testset "generated module loads and behaves" begin
        genfile = joinpath(dir, "PolicySynth.jl")
        write(genfile, code)
        M = include(genfile)
        @test M isa Module

        # Bitfield round-trip on the tail-spanning field — the old container
        # accessor read/wrote past the 8-byte struct here
        s = M.BitTail()
        s2 = M.set_x(s, 0x1ffff)
        @test M.get_x(s2) === UInt32(0x1ffff)
        @test M.get_hdr(s2) === UInt32(0)          # neighbors untouched
        s3 = M.set_x(s2, 0)
        @test M.get_x(s3) === UInt32(0)
        @test s3._data == s._data                  # full round-trip, no residue
        # Negative values wrap (old code threw InexactError)
        @test M.get_x(M.set_x(s, -1)) === UInt32(0x1ffff)

        # Blob getproperty path still works (PackedInt stays a blob)
        p = M.PackedInt()
        @test p.val === Int32(0)

        # The trap fires before any ccall
        @test_throws ErrorException M.use_packed(p)

        # Raw _ptr variants are real callables
        @test isempty(methods(M.get_msg_ptr)) == false
    end
end
