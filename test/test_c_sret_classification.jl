# =============================================================================
# Tier 3 by-value record classification (library-free, CI-safe)
#
# Where C returns a record decides the shape of the ccall GeneratorC emits: C
# returns it in registers or through a hidden pointer, and where Julia's own
# lowering of the emitted type would not pass that pointer, the wrapper passes
# it explicitly (BRANCH 0). Getting it wrong in either direction shifts every
# argument and returns garbage, with no error.
#
# C1 (abi_fuzz, seeds 3 and 36): `{ char m1; short m2[3]; }` is 8 bytes and
# comes back in RAX on both ABIs, but the generator took each member's SIZE as
# its alignment, judged the 6-byte array at offset 2 "unaligned", and passed a
# hidden pointer the callee never reads. The same test sat in the blob-param
# trap and the blob-return guard, and none of the three knew Win64, which
# classifies by size alone.
#
# Every answer is checked for BOTH conventions (`abi = :sysv` / `:win64`), on
# any host. The DWARF tables below are in the shape the parser writes: hex
# offsets, `size = 0` for record and enum members, bitfields with no offset.
# What only a real SysV callee can show (Julia's own lowering of these types)
# needs a Linux host; test_c_sret_abi.jl (devtests) runs the live calls here
# and checks this classifier against clang's lowering for both targets.
# =============================================================================

using Test
using RepliBuild
using Libdl

const _SW = RepliBuild.Wrapper

_sm(name, jt, ct, off, size) = Dict{String,Any}("name" => name, "julia_type" => jt, "c_type" => ct,
                                                "offset" => off, "size" => size)
_sbit(name, jt, ct, bits, at) = Dict{String,Any}("name" => name, "julia_type" => jt, "c_type" => ct,
                                                 "offset" => nothing, "size" => 4,
                                                 "bit_size" => bits, "data_bit_offset" => at)
_srec(size, members...; kind = "struct") =
    Dict{String,Any}("kind" => kind, "byte_size" => "0x" * string(size, base = 16), "members" => Any[members...])

const _SRET_STRUCTS = Dict{String,Any}(
    # C1, seed 36: 8 bytes, `short[3]` 2-aligned at offset 2. RAX on both ABIs.
    "T4" => _srec(8, _sm("m1", "UInt8", "char", "0x0", 1),
                     _sm("m2", "NTuple{3, Cshort}", "short[3]", "0x2", 6)),
    "R5" => _srec(4, _sm("m1", "Cint", "int", "0x0", 4)),
    # 16 bytes, `char[12]` 1-aligned at 4: two INTEGER eightbytes on SysV, sret
    # on Win64 (not 1/2/4/8). The size test called it unaligned (12 → align 8).
    "IC12" => _srec(16, _sm("n", "Cint", "int", "0x0", 4),
                        _sm("s", "NTuple{12, UInt8}", "char[12]", "0x4", 12)),
    # Packed, unaligned member, 4 and 8 bytes: RAX on Win64, MEMORY on SysV.
    "P4" => _srec(4, _sm("c", "UInt8", "char", "0x0", 1),
                     _sm("s", "Cshort", "short", "0x1", 2),
                     _sm("d", "UInt8", "char", "0x3", 1)),
    "P8" => _srec(8, _sm("c", "UInt8", "char", "0x0", 1),
                     _sm("i", "Cint", "int", "0x1", 4),
                     _sm("pad", "NTuple{3, UInt8}", "char[3]", "0x5", 3)),
    # Over 16 bytes: memory on both. Big24 and IS20 are natural, so the plain
    # ccall's own lowering passes the hidden pointer (the old size test called
    # IS20 packed: `short[8]` as align 8). PK17 is packed, so BRANCH 0 does.
    "Big24" => _srec(24, _sm("a", "Clonglong", "long long", "0x0", 8),
                         _sm("b", "Clonglong", "long long", "0x8", 8),
                         _sm("c", "Cint", "int", "0x10", 4)),
    "IS20" => _srec(20, _sm("n", "Cint", "int", "0x0", 4),
                        _sm("s", "NTuple{8, Cshort}", "short[8]", "0x4", 16)),
    "PK17" => _srec(17, _sm("c", "UInt8", "char", "0x0", 1),
                        _sm("x", "Clonglong", "long long", "0x1", 8),
                        _sm("y", "Clonglong", "long long", "0x9", 8)),
    # An unaligned field inside a packed member record. `In3` is align 1, so
    # `Out6` lays out naturally and the size test (3 + 3 bytes, 3-aligned)
    # agrees with DWARF: `is_c_lto_safe` passes it. `y` is at absolute 1.
    "In3" => _srec(3, _sm("x", "UInt8", "char", "0x0", 1),
                      _sm("y", "Cshort", "short", "0x1", 2)),
    "Out6" => _srec(6, _sm("in", "In3", "In3", "0x0", 0),
                       _sm("c", "NTuple{3, UInt8}", "char[3]", "0x3", 3)),
    # An array of a packed record: element 0's `y` lands at 2 (aligned),
    # element 1's at 5. Every element has to be walked.
    "ArrIn" => _srec(7, _sm("c", "UInt8", "char", "0x0", 1),
                        _sm("v", "NTuple{2, In3}", "In3[2]", "0x1", 0)),
    # Enum members carry size 0; E2 is a 2-byte enum.
    "EnumOk" => _srec(4, _sm("c", "UInt8", "char", "0x0", 1),
                         _sm("e", "E2", "E2", "0x2", 0)),
    "EnumPacked" => _srec(3, _sm("c", "UInt8", "char", "0x0", 1),
                             _sm("e", "E2", "E2", "0x1", 0)),
    "__enum__E2" => Dict{String,Any}("kind" => "enum", "byte_size" => "0x2", "julia_type" => "Cushort",
                                     "underlying_type" => "unsigned short",
                                     "enumerators" => Any[Dict{String,Any}("name" => "E2_A", "value" => 0)]),
    # Bitfields are skipped, as clang skips them for this rule.
    "BitOnly" => _srec(4, _sm("c", "UInt8", "char", "0x0", 1), _sbit("b", "Cint", "int", 7, 9)),
    # Float-bearing bitfield records are blobs whose SysV register class is
    # unknowable: 12 bytes (sret on Win64) and 8 bytes (RAX on Win64).
    "BF12" => _srec(12, _sm("f", "Cfloat", "float", "0x0", 4),
                        _sbit("b", "Cint", "int", 3, 32),
                        _sm("c", "Cint", "int", "0x8", 4)),
    "BF8" => _srec(8, _sm("f", "Cfloat", "float", "0x0", 4), _sbit("b", "Cint", "int", 3, 32)),
    # Members at offset 0 of a union.
    "U3" => _srec(4, _sm("i", "Cint", "int", "0x0", 4), _sm("c", "NTuple{3, UInt8}", "char[3]", "0x0", 3);
                  kind = "union"),
)

@testset "Tier 3 by-value record classification (C1)" begin

    @testset "the target comes from the host, in one place" begin
        @test _SW._c_record_abi() === (Sys.iswindows() ? :win64 : :sysv)
        @test_throws ArgumentError _SW._c_return_in_memory(8, false; abi = :aapcs64)
    end

    @testset "unaligned fields: the type's alignment, never the member's size" begin
        unaligned(k) = _SW._c_has_unaligned_field(_SRET_STRUCTS[k], _SRET_STRUCTS)
        expect = Dict("T4" => false, "R5" => false, "IC12" => false, "P4" => true, "P8" => true,
                      "Big24" => false, "IS20" => false, "PK17" => true, "In3" => true, "Out6" => true, "ArrIn" => true,
                      "EnumOk" => false, "EnumPacked" => true, "BitOnly" => false,
                      "BF12" => false, "BF8" => false, "U3" => false)
        checked = 0
        for (k, want) in sort(collect(expect))
            got = unaligned(k)
            @test (k, got) == (k, want)
            checked += 1
        end
        @test checked == length(expect)
        # ArrIn's first element is aligned; only the walk over element 1 finds `y`.
        @test _SW._c_has_unaligned_field(_srec(4, _sm("c", "UInt8", "char", "0x0", 1),
                                                  _sm("v", "NTuple{1, In3}", "In3[1]", "0x1", 0)),
                                         _SRET_STRUCTS) == false
        # An alignment nothing can derive is reported, not guessed.
        @test _SW._c_has_unaligned_field(_srec(8, _sm("q", "Any", "Mystery", "0x2", 0)),
                                         _SRET_STRUCTS) === nothing
    end

    @testset "return class per target" begin
        cls(k, abi) = let info = _SRET_STRUCTS[k]
            _SW._c_return_in_memory(_SW._parse_dwarf_size(info),
                                    _SW._c_has_unaligned_field(info, _SRET_STRUCTS); abi)
        end
        #          record      SysV   Win64
        table = [("T4",       false, false),   # C1: RAX on both
                 ("IC12",     false, true),    # INTEGER,INTEGER vs not 1/2/4/8
                 ("P4",       true,  false),   # packed 4: MEMORY vs RAX
                 ("P8",       true,  false),   # packed 8: MEMORY vs RAX
                 ("Big24",    true,  true),    # > 16
                 ("IS20",     true,  true),
                 ("PK17",     true,  true),
                 ("Out6",     true,  true),    # nested unaligned; 6 bytes
                 ("ArrIn",    true,  true),
                 ("EnumOk",   false, false),
                 ("EnumPacked", true, true),   # 3 bytes on Win64
                 ("BF12",     false, true),
                 ("BF8",      false, false),
                 ("R5",       false, false)]
        for (k, sysv, win64) in table
            @test (k, :sysv, cls(k, :sysv)) == (k, :sysv, sysv)
            @test (k, :win64, cls(k, :win64)) == (k, :win64, win64)
        end
        # Win64 never reads the alignment, SysV only at 16 bytes or less.
        @test _SW._c_return_in_memory(8, nothing; abi = :win64) == false
        @test _SW._c_return_in_memory(24, nothing; abi = :sysv) == true
        @test _SW._c_return_in_memory(8, nothing; abi = :sysv) === nothing
    end

    @testset "explicit sret only where Julia's lowering would not pass the pointer" begin
        explicit(k, abi) = let info = _SRET_STRUCTS[k]
            _SW._c_explicit_sret(_SW._parse_dwarf_size(info),
                                 _SW._c_has_unaligned_field(info, _SRET_STRUCTS); abi)
        end
        # Win64 lowers by size on both sides, so at 16 bytes or less Julia passes
        # the hidden pointer itself; SysV's unaligned rule is the one it misses.
        #          record      SysV   Win64
        table = [("T4",       false, false),
                 ("IC12",     false, false),   # Win64 sret, but Julia's own
                 ("P4",       true,  false),
                 ("P8",       true,  false),
                 ("Big24",    true,  true),    # > 16: explicit, as it always was
                 ("PK17",     true,  true),
                 ("Out6",     true,  false),
                 ("EnumPacked", true, false),
                 ("BF12",     false, false)]
        for (k, sysv, win64) in table
            @test (k, :sysv, explicit(k, :sysv)) == (k, :sysv, sysv)
            @test (k, :win64, explicit(k, :win64)) == (k, :win64, win64)
        end
        @test !_SW._c_explicit_sret(8, nothing; abi = :sysv)           # unprovable: not in memory
    end

    @testset "blob parameter: SysV MEMORY vs blob INTEGER; Win64 by size on both" begin
        mis(k, abi) = let info = _SRET_STRUCTS[k]
            _SW._c_blob_param_misclassified(_SW._parse_dwarf_size(info),
                                            _SW._c_has_unaligned_field(info, _SRET_STRUCTS); abi)
        end
        table = [("P4", true), ("P8", true), ("Out6", true), ("EnumPacked", true),
                 ("T4", false), ("IC12", false), ("BF12", false), ("Big24", false)]
        for (k, sysv) in table
            @test (k, :sysv, mis(k, :sysv)) == (k, :sysv, sysv)
            @test (k, :win64, mis(k, :win64)) == (k, :win64, false)
        end
        @test _SW._c_blob_param_misclassified(8, nothing; abi = :sysv)    # unprovable: refused
        @test !_SW._c_blob_param_misclassified(24, true; abi = :sysv)     # MEMORY on both views
        @test_throws ArgumentError _SW._c_blob_param_misclassified(8, false; abi = :aapcs64)
    end

    # ── What the generator emits for each shape, per target ──────────────────
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "sretsynth"
    root = "$(escape_string(dir))"
    [link]
    enable_lto = false
    [wrap]
    language = "c"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
    rec_param(n, t) = Dict{String,Any}("name" => n, "julia_type" => t, "c_type" => t)
    fn(name, ret, params...) = Dict{String,Any}("name" => name, "mangled" => name, "demangled" => name,
        "parameters" => Any[params...],
        "return_type" => Dict{String,Any}("julia_type" => ret, "c_type" => ret))
    meta = Dict{String,Any}(
        "struct_definitions" => _SRET_STRUCTS,
        "globals" => Dict{String,Any}(),
        "functions" => Any[
            fn("f4", "T4", rec_param("p1", "R5")),
            fn("make_ic12", "IC12", rec_param("n", "Cint")),
            fn("make_p4", "P4"),
            fn("make_p8", "P8"),
            fn("make_big24", "Big24"),
            fn("make_is20", "IS20"),
            fn("make_pk17", "PK17"),
            fn("make_out6", "Out6"),
            fn("make_bf12", "BF12"),
            fn("make_bf8", "BF8"),
            fn("take_p8", "void", rec_param("v", "P8")),
            fn("take_t4", "void", rec_param("v", "T4")),
        ])
    libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    gen(abi) = _SW.generate_introspective_module_c(cfg, libref, meta, "SretSynth",
                                                   _SW.create_type_registry(cfg), false; abi)
    body(code, name) = let i = findfirst("function $name(", code)
        i === nothing ? "" : (r = SubString(code, first(i)); j = findfirst(r"\nend\b", r);
                              j === nothing ? String(r) : String(r[1:last(j)]))
    end
    form(code, name) = let b = body(code, name)
        isempty(b) ? :missing :
        occursin("ABI Safety Trap", b) ? :trap :
        occursin("ret_buf", b) ? :sret : occursin("ccall((:$name, LIBRARY_PATH)", b) ? :ccall : :other
    end

    @test_throws ArgumentError gen(:aapcs64)
    sysv, win64 = gen(:sysv), gen(:win64)
    # :ccall on a record C returns through a hidden pointer means Julia's own
    # lowering passes it (Win64 by size; anything above 16 bytes).
    #          function      SysV     Win64
    table = [("f4",         :ccall,  :ccall),   # C1: was :sret on both
             ("make_ic12",  :ccall,  :ccall),   # was :sret on both; registers on SysV
             ("make_p4",    :sret,   :ccall),   # packed 4-byte return in RAX on Win64
             ("make_p8",    :sret,   :ccall),
             ("make_big24", :ccall,  :ccall),   # passes the size test; Julia's own sret
             ("make_is20",  :ccall,  :ccall),   # natural, > 16: Julia's own sret (the size test called it packed)
             ("make_pk17",  :sret,   :sret),    # packed, > 16
             ("make_out6",  :sret,   :ccall),   # SysV: nested unaligned field, size test blind
             ("make_bf12",  :trap,   :ccall),   # float blob: SysV class unknowable; Win64 sret by size
             ("take_p8",    :trap,   :ccall),   # SysV MEMORY vs blob INTEGER; Win64 i64 both
             ("take_t4",    :ccall,  :ccall)]
    checked = 0
    for (name, want_sysv, want_win64) in table
        @test (name, :sysv, form(sysv, name)) == (name, :sysv, want_sysv)
        @test (name, :win64, form(win64, name)) == (name, :win64, want_win64)
        checked += 1
    end
    @test checked == length(table)
    # A float blob in RAX stays refused on SysV, whose register class it cannot
    # know. (Win64 needs no class, but the guard is SysV-shaped; not pinned.)
    @test form(sysv, "make_bf8") == :trap
    # The plain ccall returns the record itself, with no hidden first argument.
    @test occursin("ccall((:f4, LIBRARY_PATH), T4, (R5,), p1)", body(sysv, "f4"))
    @test occursin("ccall((:f4, LIBRARY_PATH), T4, (R5,), p1)", body(win64, "f4"))
    @test occursin("ccall((:make_p4, LIBRARY_PATH), Cvoid, (Ptr{P4}, ), ret_buf, )", body(sysv, "make_p4"))
    @test occursin("ccall((:make_ic12, LIBRARY_PATH), IC12, (Cint,), n_c)", body(win64, "make_ic12"))
    @test occursin("ccall((:make_is20, LIBRARY_PATH), IS20, (), )", body(win64, "make_is20"))
    @test occursin("ccall((:make_pk17, LIBRARY_PATH), Cvoid, (Ptr{PK17}, ), ret_buf, )", body(win64, "make_pk17"))
end
