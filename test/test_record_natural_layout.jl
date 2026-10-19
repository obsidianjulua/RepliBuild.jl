# =============================================================================
# Is a record's layout natural? (library-free, CI-safe)
#
# `_record_natural_layout` answers "does a Julia struct of these member types
# reproduce the C layout?" for C's `is_c_lto_safe` and for C++'s
# `_is_struct_unsafe`, which decides Tier 3 (ccall) against Tier 2 (thunk) in
# `is_ccall_safe`. The old answer laid the members out again with each member's
# SIZE as its alignment, capped at 8, and compared sizes. That called
# `{ char; short[3]; }`, `{ float; Vec2; float; }` and every record with an enum
# member (size 0 in the metadata) packed, and sent them to Tier 2. Its nested
# check looked one level down with no type table, so a packed record two levels
# down reached ccall.
#
# GeneratorCpp also sends a function that crosses a byte blob of 16 bytes or
# less to Tier 2: a natural record the struct emitter could not prove (an enum
# member with size 0 makes one) is a blob, and a blob claims INTEGER for every
# SysV eightbyte.
# =============================================================================

using Test
using RepliBuild
using Libdl

const _NW = RepliBuild.Wrapper

_nm(name, jt, ct, off, size) = Dict{String,Any}("name" => name, "julia_type" => jt, "c_type" => ct,
                                                "offset" => off, "size" => size)
_nrec(size, members...; kind = "struct") =
    Dict{String,Any}("kind" => kind, "byte_size" => "0x" * string(size, base = 16), "members" => Any[members...])

const _NATURAL_STRUCTS = Dict{String,Any}(
    # Natural, and packed by the size test
    "CS3" => _nrec(8, _nm("c", "UInt8", "char", "0x0", 1), _nm("s", "NTuple{3, Cshort}", "short[3]", "0x2", 6)),
    "Vec2" => _nrec(8, _nm("x", "Cfloat", "float", "0x0", 4), _nm("y", "Cfloat", "float", "0x4", 4)),
    "Mass" => _nrec(16, _nm("m", "Cfloat", "float", "0x0", 4), _nm("c", "Vec2", "Vec2", "0x4", 8),
                        _nm("i", "Cfloat", "float", "0xc", 4)),
    "IF3" => _nrec(16, _nm("n", "Cint", "int", "0x0", 4), _nm("v", "NTuple{3, Cfloat}", "float[3]", "0x4", 12)),
    # The enum member with size 0 is metadata written before `parse_dwarf_dump`
    # sized enum members, which a package keeps until it is rebuilt: the C++
    # struct emitter cannot prove it and writes a blob. EnumRecSized is what the
    # parser writes now.
    "EnumRec" => _nrec(4, _nm("c", "UInt8", "char", "0x0", 1), _nm("e", "E2", "E2", "0x2", 0)),
    "EnumRecSized" => _nrec(4, _nm("c", "UInt8", "char", "0x0", 1), _nm("e", "E2", "E2", "0x2", 2)),
    "BigEnum" => _nrec(24, _nm("e", "E2", "E2", "0x0", 0), _nm("d1", "Cdouble", "double", "0x8", 8),
                           _nm("d2", "Cdouble", "double", "0x10", 8)),
    "__enum__E2" => Dict{String,Any}("kind" => "enum", "byte_size" => "0x2", "julia_type" => "Cushort",
                                     "underlying_type" => "unsigned short",
                                     "enumerators" => Any[Dict{String,Any}("name" => "E2_A", "value" => 0)]),
    # Natural either way
    "Wide" => _nrec(16, _nm("d", "Cdouble", "double", "0x0", 8), _nm("f", "Cfloat", "float", "0x8", 4)),
    "ArrV" => _nrec(16, _nm("v", "NTuple{2, Vec2}", "Vec2[2]", "0x0", 16)),
    "WithStatic" => _nrec(4, _nm("a", "Cint", "int", "0x0", 4), _nm("s", "Cdouble", "double", nothing, 8)),
    "Empty" => _nrec(1),
    # Not natural
    "Pk5" => _nrec(5, _nm("c", "UInt8", "char", "0x0", 1), _nm("i", "Cint", "int", "0x1", 4)),
    "Pr6" => _nrec(6, _nm("c", "UInt8", "char", "0x0", 1), _nm("i", "Cint", "int", "0x2", 4)),
    "In3" => _nrec(3, _nm("x", "UInt8", "char", "0x0", 1), _nm("y", "Cshort", "short", "0x1", 2)),
    # The sizes line up at every level, so only a walk into In3 finds it.
    "Out6" => _nrec(6, _nm("in", "In3", "In3", "0x0", 3), _nm("c", "NTuple{3, UInt8}", "char[3]", "0x3", 3)),
    "Top6" => _nrec(6, _nm("o", "Out6", "Out6", "0x0", 6)),
    "Gap12" => _nrec(12, _nm("c", "UInt8", "char", "0x0", 1), _nm("i", "Cint", "int", "0x8", 4)),
    "BitRec" => _nrec(4, _nm("c", "UInt8", "char", "0x0", 1),
                         Dict{String,Any}("name" => "b", "julia_type" => "Cint", "c_type" => "int",
                                          "offset" => nothing, "size" => 4, "bit_size" => 3,
                                          "data_bit_offset" => 8)),
    "U4" => _nrec(4, _nm("i", "Cint", "int", "0x0", 4), _nm("f", "Cfloat", "float", "0x0", 4); kind = "union"),
    "HoldsU" => _nrec(8, _nm("u", "U4", "U4", "0x0", 4), _nm("k", "Cint", "int", "0x4", 4)),
    # Unmeasurable
    "Mystery" => _nrec(8, _nm("c", "UInt8", "char", "0x0", 1), _nm("q", "Any", "Unknowable", "0x4", 0)),
    # What the C++ struct emitter's field path meets once members are sized:
    # a static member (no offset), an array of a template record, an array of a
    # type the module never emits, members whose type is never emitted (`pair<`
    # is STL-internal) and so become bytes, and a field whose record is a blob.
    "WithStaticRec" => _nrec(8, _nm("count", "Cint", "int", nothing, 4), _nm("a", "Cint", "int", "0x0", 4),
                                _nm("b", "Cfloat", "float", "0x4", 4)),
    "Duo<int>" => _nrec(8, _nm("x", "Cint", "int", "0x0", 4), _nm("y", "Cint", "int", "0x4", 4)),
    "TplArr" => _nrec(16, _nm("v", "NTuple{2, Duo<int>}", "Duo<int>[2]", "0x0", 16)),
    "GhostArr" => _nrec(16, _nm("g", "NTuple{2, Ghost}", "Ghost[2]", "0x0", 16)),
    "pair<int, int>" => _nrec(8, _nm("first", "Cint", "int", "0x0", 4), _nm("second", "Cint", "int", "0x4", 4)),
    "pair<float, float>" => _nrec(8, _nm("first", "Cfloat", "float", "0x0", 4),
                                     _nm("second", "Cfloat", "float", "0x4", 4)),
    "IntPairRec" => _nrec(8, _nm("p", "pair<int, int>", "std::pair<int, int>", "0x0", 8)),
    "FloatPairRec" => _nrec(8, _nm("p", "pair<float, float>", "std::pair<float, float>", "0x0", 8)),
    # A blob (its enum member has size 0) holding a float, as a field.
    "FBlob" => _nrec(8, _nm("x", "Cfloat", "float", "0x0", 4), _nm("e", "E2", "E2", "0x4", 0)),
    "HoldsFBlob" => _nrec(8, _nm("v", "FBlob", "FBlob", "0x0", 8)),
    # A bitfield record is a `mutable struct`: as a field, a reference.
    "BitHolder" => _nrec(8, _nm("b", "BitRec", "BitRec", "0x0", 4), _nm("k", "Cint", "int", "0x4", 4)),
    # Spelled as its enum, sized as the pointer it is (fmt's `arg_id_kind&`).
    "RefEnumRec" => _nrec(8, _nm("r", "E2", "E2&", "0x0", 8)),
    # An enum whose name is a Base binding is emitted as `c_Type`; a function
    # taking it by value asks for a forward declaration of the same name.
    "__enum__Type" => Dict{String,Any}("kind" => "enum", "byte_size" => "0x4", "julia_type" => "Cuint",
                                       "underlying_type" => "unsigned int",
                                       "enumerators" => Any[Dict{String,Any}("name" => "TY_A", "value" => 0),
                                                            Dict{String,Any}("name" => "TY_B", "value" => 1)]),
    "HasType" => _nrec(8, _nm("t", "Type", "Type", "0x0", 4), _nm("k", "Cint", "int", "0x4", 4)),
    # A named union is `mutable`, a reference as a field. At 8 bytes and an
    # aligned offset it fits the proof's pointer measure.
    "U8" => _nrec(8, _nm("i", "Clonglong", "long long", "0x0", 8), _nm("d", "Cdouble", "double", "0x0", 8);
                  kind = "union"),
    "HoldsU8" => _nrec(16, _nm("k", "Cint", "int", "0x0", 4), _nm("u", "U8", "U8", "0x8", 8)),
)

@testset "Record layouts: natural or not (routing)" begin

    @testset "natural layout from the member types' alignment" begin
        natural(k) = _NW._record_natural_layout(_NATURAL_STRUCTS[k], _NATURAL_STRUCTS)
        expect = Dict("CS3" => true, "Vec2" => true, "Mass" => true, "IF3" => true, "EnumRec" => true,
                      "EnumRecSized" => true,
                      "BigEnum" => true, "Wide" => true, "ArrV" => true, "WithStatic" => true,
                      "Empty" => true, "WithStaticRec" => true, "TplArr" => true, "IntPairRec" => true,
                      "FloatPairRec" => true, "HoldsFBlob" => true,
                      "Pk5" => false, "Pr6" => false, "In3" => false, "Out6" => false, "Top6" => false,
                      "Gap12" => false, "BitRec" => false, "U4" => false, "HoldsU" => false,
                      "Mystery" => nothing)
        checked = 0
        for (k, want) in sort(collect(expect); by = first)
            @test (k, natural(k)) === (k, want)
            checked += 1
        end
        @test checked == length(expect)
    end

    # extern "C" functions (unmangled): no exception routing, so the record decides.
    fn(name, ret, params...) = Dict{String,Any}("name" => name, "mangled" => name, "demangled" => name,
        "parameters" => Any[Dict{String,Any}("name" => "p$i", "julia_type" => t, "c_type" => t)
                            for (i, t) in enumerate(params)],
        "return_type" => Dict{String,Any}("julia_type" => ret == "void" ? "Cvoid" : ret, "c_type" => ret))

    @testset "C++ routing: is_ccall_safe" begin
        safe(rec) = (_NW.is_ccall_safe(fn("take", "void", rec), _NATURAL_STRUCTS),
                     _NW.is_ccall_safe(fn("give", rec), _NATURAL_STRUCTS))
        # Tier 3 now; the size test sent each to Tier 2
        for k in ("CS3", "Mass", "IF3", "EnumRec")
            @test (k, safe(k)) == (k, (true, true))
        end
        # Tier 2: not natural, at any depth. Top6 reached ccall before: the old
        # nested check looked at Out6 with no type table and could not see In3.
        for k in ("Pk5", "Pr6", "Out6", "Top6", "Gap12", "BitRec", "HoldsU", "U4", "Mystery")
            @test (k, safe(k)) == (k, (false, false))
        end
    end

    @testset "C: is_c_lto_safe" begin
        lto(rec) = _NW.is_c_lto_safe(fn("give", rec), _NATURAL_STRUCTS)
        for (k, want) in (("CS3", true), ("IF3", true), ("BigEnum", true),
                          ("Pk5", false), ("Top6", false), ("U4", false), ("Mystery", false))
            @test (k, lto(k)) == (k, want)
        end
    end

    @testset "GeneratorCpp: a ≤16-byte blob does not ride a ccall" begin
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "naturalsynth"
        root = "$(escape_string(dir))"
        [wrap]
        language = "cpp"
        [types]
        strictness = "warn"
        allow_unknown_structs = true
        [cache]
        enabled = false
        """)
        cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
        meta = Dict{String,Any}(
            "struct_definitions" => _NATURAL_STRUCTS,
            "record_abi" => Dict{String,Any}(),
            "globals" => Dict{String,Any}(),
            "functions" => Any[fn("take_cs3", "void", "CS3"),         # exact Julia struct
                               fn("take_enumrec", "void", "EnumRec"), # natural, emitted as a 4-byte blob
                               fn("take_enumrec_sized", "void", "EnumRecSized"), # the same, sized: fields
                               fn("take_bigenum", "void", "BigEnum"), # natural, a 24-byte blob
                               fn("take_pk5", "void", "Pk5"),         # packed
                               fn("take_static", "void", "WithStaticRec"),
                               fn("take_tplarr", "void", "TplArr"),
                               fn("take_intpair", "void", "IntPairRec"),     # bytes over two ints
                               fn("take_floatpair", "void", "FloatPairRec"), # bytes over two floats
                               fn("take_holdsfblob", "void", "HoldsFBlob"), # a blob over a float, as a field
                               fn("take_type", "void", "Type")])
        libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
        res = _NW.generate_introspective_module_cpp(cfg, libref, meta, "NaturalSynth",
                                                    _NW.create_type_registry(cfg), false)
        code = res isa Tuple ? res[1] : res
        # The two records this hinges on really are blobs here.
        @test occursin(r"struct EnumRec\n\s+_data::NTuple\{4, UInt8\}", code)
        @test occursin(r"struct BigEnum\n\s+_data::NTuple\{24, UInt8\}", code)
        @test occursin(r"struct EnumRecSized\n\s+c::UInt8\n\s+_pad_0::NTuple\{1, UInt8\}\n\s+e::E2\n", code)

        @testset "fields for what the parser now sizes" begin
            # The static member is not a field (it was one at offset 0, which
            # made every record with one a blob).
            @test occursin(r"struct WithStaticRec\n\s+a::Cint\n\s+b::Cfloat\nend", code)
            # An array of a template record names the element as emitted; it
            # was sanitized whole into `NTuple_2_Duo_int`, which nothing declares.
            @test occursin(r"struct TplArr\n\s+v::NTuple\{2, Duo_int\}\nend", code)
            # Element and member types the module never emits become their bytes.
            @test occursin(r"struct GhostArr\n\s+g::NTuple\{16, UInt8\}\nend", code)
            @test occursin(r"struct FloatPairRec\n\s+p::NTuple\{8, UInt8\}\nend", code)
            @test occursin(r"struct HoldsFBlob\n\s+v::FBlob\nend", code)
            # A field type the layout proof cannot measure is bytes too: as
            # `b::BitRec` it was a reference, unproven, and 16 bytes for 8.
            @test occursin(r"struct BitHolder\n\s+b::NTuple\{4, UInt8\}\n\s+k::Cint\nend", code)
            # A type of another size than the member's is not the member's type.
            @test occursin(r"struct RefEnumRec\n\s+r::NTuple\{8, UInt8\}\nend", code)
            # No forward declaration replaces an enum: `struct c_Type end` after
            # `@enum c_Type` made every later `::c_Type` an empty struct.
            @test occursin(r"\n@enum c_Type::Cuint", code)
            @test !occursin(r"\nstruct c_Type end", code)
            @test occursin(r"struct HasType\n\s+t::c_Type\n\s+k::Cint\nend", code)
            # A record holding a named union is a blob, whatever the sizes say.
            @test occursin(r"struct HoldsU8\n\s+_data::NTuple\{16, UInt8\}\nend", code)
        end

        tiers = Dict(Symbol(m.captures[1]) => Symbol(m.captures[2])
                     for m in eachmatch(r"\n\s+:(\w+) => :(tier\d)", code))
        want = Dict(:take_cs3 => :tier3, :take_enumrec => :tier2, :take_enumrec_sized => :tier3,
                    :take_bigenum => :tier3, :take_pk5 => :tier2,
                    :take_static => :tier3, :take_tplarr => :tier3,
                    # Bytes where C has a float or double ride the thunk at
                    # ≤16 bytes, as a blob does; over integers they need not.
                    :take_intpair => :tier3, :take_floatpair => :tier2, :take_holdsfblob => :tier2)
        checked = 0
        for (f, t) in sort(collect(want))
            @test (f, get(tiers, f, :missing)) == (f, t)
            checked += 1
        end
        @test checked == length(want)

        @testset "the module loads; sizes, enum field, setters" begin
            host = Module(:NaturalSynthHost)
            Base.include_string(host, code)
            W = getfield(host, :NaturalSynth)
            sizes = Dict(:WithStaticRec => 8, :TplArr => 16, :GhostArr => 16, :IntPairRec => 8,
                         :FloatPairRec => 8, :HoldsFBlob => 8, :EnumRecSized => 4, :CS3 => 8, :Mass => 16,
                         :BitHolder => 8, :RefEnumRec => 8, :HasType => 8, :HoldsU8 => 16)
            checked = 0
            for (s, n) in sort(collect(sizes))
                @test (s, sizeof(getfield(W, s))) == (s, n)
                checked += 1
            end
            @test checked == length(sizes)
            # Every struct emitted immutable holds C's bytes and nothing else: a
            # reference field is a pointer the GC follows into those bytes.
            immut = [Symbol(m.captures[1]) for m in eachmatch(r"\nstruct (\w+)\n", code)]
            @test length(immut) >= length(sizes)
            notbits = [s for s in immut if isdefined(W, s) && !isbitstype(getfield(W, s))]
            @test notbits == Symbol[]
            @test fieldtype(W.EnumRecSized, :e) === W.E2
            @test W.EnumRecSized().e === W.E2_A
            @test W.c_Type <: Enum && fieldtype(W.HasType, :t) === W.c_Type
            # The setters a blob had, on a struct with fields.
            x = W.WithStaticRec()
            y = W.setproperty(x, :b, 2.5)
            @test (y.b, x.b) === (2.5f0, 0.0f0)
            @test W.setproperties(x; a = 3, b = 1.5) === W.WithStaticRec(Cint(3), 1.5f0)
            @test W.setproperty(W.EnumRecSized(), :e, W.E2_A).e === W.E2_A
            @test_throws ErrorException W.setproperty(x, :not_a_field, 1)
            err = try (z = W.WithStaticRec(); z.b = 1; "") catch e sprint(showerror, e) end
            @test occursin("setproperties", err)
        end
    end
end
