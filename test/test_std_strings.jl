# test/test_std_strings.jl — a std::string returned by value is a Julia String
# (2026-11-13). Toolchain-free: libstdc++- and libc++-shaped metadata and bytes.
#
# The build records each by-value basic_string's layout (`Compiler._string_layout`:
# leaves with dotted paths and absolute offsets, the character type, the declared
# destructor) on its `record_abi` entry. `StdStrings.string_reader` turns that
# into a `Reader` for the two layouts that exist, and refuses anything else with
# a reason; GeneratorCpp's `_string_return_plan` decides per function, and
# `StdStrings.take_string!` reads the object where C++ built it and ends it.
# The live calls are devtests' test_sret_in_caller_buffer.jl.

using Test
using RepliBuild

const _SS = RepliBuild.StdStrings
const _SW = RepliBuild.Wrapper
const _SC = RepliBuild.Compiler

_leaf(path, ct, off; size = 8) = Dict{String,Any}("path" => path, "c_type" => ct, "offset" => off, "size" => size)
_bits(path, ct, bit, n) = Dict{String,Any}("path" => path, "c_type" => ct, "bit_size" => n, "data_bit_offset" => bit)

# What the build records for libstdc++'s cxx11 string (read off clang 23's DWARF).
const _LIBSTDCXX = Dict{String,Any}("char_type" => "char",
    "destructor" => "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED4Ev",
    "leaves" => Any[_leaf("_M_dataplus._M_p", "char*", 0), _leaf("_M_string_length", "size_t", 8),
                    _leaf("_anon1._M_local_buf", "char[16]", 16; size = 16),
                    _leaf("_anon1._M_allocated_capacity", "size_t", 16)])
# libc++'s standard layout since LLVM 19: the long/short flag is bit 0, a short
# string's length the seven bits above it and its characters from byte 1; a long
# one's length at 8 and its data pointer at 16.
const _LIBCXX = Dict{String,Any}("char_type" => "char", "destructor" => "_ZNSt3__112basic_stringIcEED4Ev",
    "leaves" => Any[_bits("__rep_._anon1.__l.__is_long_", "size_t", 0, 1),
                    _bits("__rep_._anon1.__l.__cap_", "size_t", 1, 63),
                    _leaf("__rep_._anon1.__l.__size_", "size_t", 8),
                    _leaf("__rep_._anon1.__l.__data_", "char*", 16),
                    _bits("__rep_._anon1.__s.__is_long_", "unsigned char", 0, 1),
                    _bits("__rep_._anon1.__s.__size_", "unsigned char", 1, 7),
                    _leaf("__rep_._anon1.__s.__data_", "char[23]", 1; size = 23)])

@testset "std::string returned by value is a Julia String" begin

@testset "the layout reader: two layouts, refusals that say why" begin
    r = _SS.string_reader(_LIBSTDCXX)
    @test r isa _SS.Reader
    @test (r.kind, r.ptr, r.len) == (:pointer_length, 0, 8)
    c = _SS.string_reader(_LIBCXX)
    @test c isa _SS.Reader
    @test (c.kind, c.ptr, c.len, c.flag_bit, c.short_data, c.short_len_bit, c.short_len_bits) ==
          (:short_long, 16, 8, 0, 1, 1, 7)
    # Refused, each with its reason, never guessed.
    @test occursin("character type is `wchar_t`", _SS.string_reader(merge(_LIBSTDCXX, Dict("char_type" => "wchar_t"))))
    @test occursin("could not be read", _SS.string_reader(Dict{String,Any}("unread" => "virtual base `X`",
                                                                          "char_type" => "char", "leaves" => Any[])))
    old_libcxx = merge(_LIBCXX, Dict{String,Any}("leaves" => filter(l -> !endswith(l["path"], "__is_long_"), _LIBCXX["leaves"])))
    @test occursin("without the `__is_long_` bit", _SS.string_reader(old_libcxx))
    @test occursin("neither libstdc++'s",
                   _SS.string_reader(Dict{String,Any}("char_type" => "char", "leaves" => Any[_leaf("p", "char*", 0)])))
    # Ubuntu 24.04's libstdc++ 13: `_Alloc_hider` is a declaration-only DIE
    # (constructor homing), so the build leaves `_M_dataplus` a leaf. Of exactly
    # pointer size it is `_M_p`, under either spelling of its type; any other
    # size is refused with the reason.
    rest = _LIBSTDCXX["leaves"][2:end]
    homed = merge(_LIBSTDCXX, Dict{String,Any}("leaves" => Any[_leaf("_M_dataplus", "_Alloc_hider", 0), rest...]))
    h = _SS.string_reader(homed)
    @test h isa _SS.Reader
    @test (h.kind, h.ptr, h.len) == (:pointer_length, 0, 8)
    # What the parser really records for a declaration-only type: size 0 (no
    # byte_size on the DIE). The span is then the distance to the next leaf.
    sized_by_next = merge(_LIBSTDCXX, Dict{String,Any}("leaves" => Any[_leaf("_M_dataplus", "_Alloc_hider", 0; size = 0), rest...]))
    h0 = _SS.string_reader(sized_by_next)
    @test h0 isa _SS.Reader
    @test (h0.kind, h0.ptr, h0.len) == (:pointer_length, 0, 8)
    # Size 0 and the next leaf 16 bytes on: not a pointer, refused with the span.
    far = merge(_LIBSTDCXX, Dict{String,Any}("leaves" => Any[_leaf("_M_dataplus", "_Alloc_hider", 0; size = 0),
                                                            _leaf("_M_string_length", "size_t", 16), rest[2:end]...]))
    @test occursin("spans 16 bytes, not a pointer", _SS.string_reader(far))
    scoped = merge(_LIBSTDCXX, Dict{String,Any}("leaves" => Any[_leaf("_M_dataplus",
                   "basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider", 0), rest...]))
    @test _SS.string_reader(scoped) isa _SS.Reader
    wide = merge(_LIBSTDCXX, Dict{String,Any}("leaves" => Any[_leaf("_M_dataplus", "_Alloc_hider", 0; size = 16), rest...]))
    @test occursin("`_Alloc_hider` has no definition", _SS.string_reader(wide))
    @test occursin("spans 16 bytes, not a pointer", _SS.string_reader(wide))
    # A leaf of another name or type is not taken for the pointer.
    other = merge(_LIBSTDCXX, Dict{String,Any}("leaves" => Any[_leaf("_M_dataplus", "size_t", 0), rest...]))
    @test occursin("neither libstdc++'s", _SS.string_reader(other))
end

@testset "reading the characters where the object is" begin
    # libstdc++, short: `_M_p` points at the object's own local buffer.
    obj = zeros(UInt8, 32)
    GC.@preserve obj begin
        base = pointer(obj)
        unsafe_store!(Ptr{Ptr{UInt8}}(base), base + 16)
        unsafe_store!(Ptr{Csize_t}(base + 8), 5)
        copyto!(obj, 17, codeunits("hello"), 1, 5)
        @test _SS.read_string(base, _SS.string_reader(_LIBSTDCXX)) == "hello"
    end
    # libc++, short: flag 0, length 3 in bits 1..7, characters from byte 1.
    sobj = zeros(UInt8, 24)
    sobj[1] = UInt8(3 << 1)
    copyto!(sobj, 2, codeunits("abc"), 1, 3)
    GC.@preserve sobj @test _SS.read_string(pointer(sobj), _SS.string_reader(_LIBCXX)) == "abc"
    # libc++, long: flag 1, length at 8, data pointer at 16.
    heap = Vector{UInt8}(codeunits("a long string on the heap"))
    lobj = zeros(UInt8, 24)
    GC.@preserve heap lobj begin
        lobj[1] = 0x01
        unsafe_store!(Ptr{Csize_t}(pointer(lobj) + 8), length(heap))
        unsafe_store!(Ptr{Ptr{UInt8}}(pointer(lobj) + 16), pointer(heap))
        @test _SS.read_string(pointer(lobj), _SS.string_reader(_LIBCXX)) == "a long string on the heap"
    end
end

@testset "per function: decoded, or the stand-in with a reason naming the type" begin
    key = "basic_string<char, std::char_traits<char>, std::allocator<char> >"
    ra = Dict{String,Any}(key => Dict{String,Any}("pass" => "reference", "byte_size" => "0x20",
                                                  "string_layout" => _LIBSTDCXX))
    plan = _SW._string_return_plan(key, ra)
    @test plan isa Tuple
    @test plan[2] == "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev"   # D4 → D1
    @test _SW._string_return_plan("const std::$key", ra) isa Tuple                  # spelled qualified
    # Metadata from before the layout was recorded: a rebuild, said so.
    old = Dict{String,Any}(key => Dict{String,Any}("pass" => "reference", "byte_size" => "0x20"))
    why = _SW._string_return_plan(key, old)
    @test why isa String && occursin("std::string `$key` returned as its raw bytes", why) &&
          occursin("rebuild", why)
    # A wide string names its type and why.
    wkey = "basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> >"
    wra = Dict{String,Any}(wkey => Dict{String,Any}("pass" => "reference",
                                                    "string_layout" => merge(_LIBSTDCXX, Dict("char_type" => "wchar_t"))))
    @test occursin("`$wkey` returned as its raw bytes: its character type is `wchar_t`", _SW._string_return_plan(wkey, wra))
    # No destructor in the DWARF: the stand-in.
    nod = Dict{String,Any}(key => Dict{String,Any}("string_layout" => merge(_LIBSTDCXX, Dict("destructor" => nothing))))
    @test occursin("declares no destructor", _SW._string_return_plan(key, nod))
    # Not a string at all.
    @test _SW._string_return_plan("Ticket", ra) === nothing
end

# The `_string_param_plan` definition as written in GeneratorCpp.jl, for the
# negative check to put back.
function _plan_source()
    src = read(joinpath(pkgdir(RepliBuild), "src", "Wrapper", "Cpp", "GeneratorCpp.jl"), String)
    i = findfirst("function _string_param_plan(", src)
    j = findnext("\nend\n", src, last(i))
    return src[first(i):last(j)]
end

@testset "a std::string parameter takes a Julia String (2026-11-15)" begin
    key = "basic_string<char, std::char_traits<char>, std::allocator<char> >"
    ctor = "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC4EPKcmRKS3_"
    this = "$key*"

    @testset "the build picks the constructor from a pointer and a length" begin
        libstdcxx = Dict(
            ctor => [this, "const char*", "size_t", "const allocator<char>&"],
            "_ZN…C4ERKS3_" => [this, "const allocator<char>&"],                      # allocator alone
            "_ZN…C4ERKS4_" => [this, "const $key&"],                                  # copy
            "_ZN…C4EPKcRKS3_" => [this, "const char*", "const allocator<char>&"],   # no length
            "_ZN…C4EIS3_EEmcRKS3_" => [this, "size_t", "char", "const allocator<char>&"])
        pick = _SC._string_constructor(collect(keys(libstdcxx)), s -> get(libstdcxx, s, nothing))
        @test pick == (ctor, true)
        # libc++ declares the two-argument form too; it is preferred (no allocator byte).
        libcxx = merge(libstdcxx, Dict("_ZNSt3__1…C4EPKcm" => [this, "const char*", "unsigned long"]))
        @test _SC._string_constructor(collect(keys(libcxx)), s -> get(libcxx, s, nothing)) == ("_ZNSt3__1…C4EPKcm", false)
        # `char const*`, a `size_type` typedef: the same constructor.
        spelled = Dict("_ZN…C4EPKcm" => [this, "char const*", "size_type"])
        @test _SC._string_constructor(collect(keys(spelled)), s -> get(spelled, s, nothing)) == ("_ZN…C4EPKcm", false)
        # None of that shape, or no parameters known: nothing.
        @test _SC._string_constructor(["_ZN…C4ERKS4_"], s -> [this, "const $key&"]) === nothing
        @test _SC._string_constructor([ctor], s -> nothing) === nothing
        @test _SC._string_constructor(String[], s -> nothing) === nothing
        # Recorded on the layout beside the destructor.
        m(name, ct, off, size) = Dict{String,Any}("name" => name, "c_type" => ct, "offset" => off, "size" => size)
        defs = Dict{String,Dict{String,Any}}(key => Dict{String,Any}("kind" => "class", "byte_size" => "0x20",
            "template_params" => Any[Dict{String,Any}("kind" => "type", "name" => "_CharT", "type" => "char")],
            "members" => Any[m("_M_p", "char*", "0x0", 8), m("_M_string_length", "size_t", "0x8", 8)]))
        L = _SC._string_layout(key, defs, Dict{String,Any}(), "_ZN…D4Ev";
                               constructors = collect(keys(libstdcxx)), params_of = s -> get(libstdcxx, s, nothing))
        @test L["constructor"] == ctor && L["constructor_allocator"] == true
        L0 = _SC._string_layout(key, defs, Dict{String,Any}(), "_ZN…D4Ev")
        @test L0["constructor"] === nothing && L0["constructor_allocator"] === nothing
    end

    layout = merge(_LIBSTDCXX, Dict{String,Any}("constructor" => ctor, "constructor_allocator" => true))
    ra = Dict{String,Any}(key => Dict{String,Any}("pass" => "reference", "byte_size" => "0x20", "string_layout" => layout))
    c1 = "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EPKcmRKS3_"
    d1 = "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev"

    @testset "per parameter: built, passed through, or the stand-in with a reason" begin
        for t in ("const $key&", key, "const std::$key&", "std::$key", "$key const&")
            @test (t, _SW._string_param_plan(t, ra)) == (t, (key, 32, c1, true, d1))
        end
        # Not built: a pointer, a non-const reference (an out-parameter), an
        # rvalue reference, anything else.
        for t in ("$key*", "const $key*", "$key&", "$key&&", "int", "const char*", "basic_string_view<char>")
            @test (t, _SW._string_param_plan(t, ra)) == (t, nothing)
        end
        # Spelled: `C4` and `C2` become the complete-object `C1`; `C1` stays.
        @test _SW._complete_object_ctor(ctor) == c1
        @test _SW._complete_object_ctor(replace(ctor, "C4E" => "C2E")) == c1
        @test _SW._complete_object_ctor(c1) == c1
        # The stand-in, with the reason naming the type.
        old = Dict{String,Any}(key => Dict{String,Any}("pass" => "reference", "byte_size" => "0x20"))
        @test occursin("predates `string_layout`", _SW._string_param_plan("const $key&", old))
        pre = Dict{String,Any}(key => Dict{String,Any}("pass" => "reference", "byte_size" => "0x20", "string_layout" => _LIBSTDCXX))
        @test occursin("predates the recorded constructor", _SW._string_param_plan("const $key&", pre))
        nocon = Dict{String,Any}(key => Dict{String,Any}("pass" => "reference", "byte_size" => "0x20",
                                                         "string_layout" => merge(layout, Dict("constructor" => nothing))))
        @test occursin("declares no constructor", _SW._string_param_plan("const $key&", nocon))
        wkey = "basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> >"
        wra = Dict{String,Any}(wkey => Dict{String,Any}("pass" => "reference", "byte_size" => "0x20",
                                                        "string_layout" => merge(layout, Dict("char_type" => "wchar_t"))))
        @test occursin("`$wkey` takes its byte stand-in: its character type is `wchar_t`", _SW._string_param_plan("const $wkey&", wra))
    end

    @testset "StdStrings.string_arg: a pointer passes through, a missing constructor is an error" begin
        lib = Base.Libc.Libdl.dlpath(Base.Libc.Libdl.dlopen(Base.libllvm_path()))
        a = _SS.string_arg(Ptr{Cvoid}(UInt(0x1234)), 32, lib, c1, true, d1, "f")
        @test _SS.address(a) == Ptr{Cvoid}(UInt(0x1234)) && a.slot === nothing
        r0 = _SS.RELEASES[]
        _SS.release!(a)
        @test _SS.RELEASES[] == r0                       # nothing to end
        # The STL wrapper's own string object passes through too, and is kept alive.
        cs = RepliBuild.STLWrappers.CppString(Ptr{Cvoid}(UInt(0x4321)), false, Dict{String,String}())
        b = _SS.string_arg(cs, 32, lib, c1, true, d1, "f")
        @test _SS.address(b) == Ptr{Cvoid}(UInt(0x4321)) && b.slot === nothing && b.keep === cs
        _SS.release!(b)
        @test _SS.RELEASES[] == r0
        @test occursin("CppString", _SW._STD_STRING_PARAM)
        err = try
            _SS.string_arg("abc", 32, lib, "_ZN11no_such_type4noneC1EPKcmRKS_", true, "_ZN11no_such_type4noneD1Ev", "f_without_ctor")
            nothing
        catch e; e; end
        @test err isa ErrorException
        @test occursin("f_without_ctor: cannot pass a Julia string", sprint(showerror, err))
        @test occursin("_ZN11no_such_type4noneC1EPKcmRKS_", sprint(showerror, err))
    end

    @testset "the wrapper: the spelling, the body, the const char* sibling" begin
        fn(name, mangled, ps) = Dict{String,Any}("name" => name, "mangled" => mangled, "demangled" => name,
            "class" => "", "is_method" => false, "is_vararg" => false, "is_noexcept" => false,
            "has_object_pointer" => nothing,
            "parameters" => Any[Dict{String,Any}("name" => n, "c_type" => c, "julia_type" => j) for (n, c, j) in ps],
            "return_type" => Dict{String,Any}("c_type" => "int", "julia_type" => "Cint"))
        fns = Any[fn("kind_of", "_Z7kind_ofRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", [("s", "const $key&", "Ref{$key}")]),
                  fn("kind_of", "_Z7kind_ofPKc", [("s", "const char*", "Cstring")]),
                  fn("str_size_v", "_Z10str_size_vNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", [("s", key, "Any")]),
                  fn("two", "_Z3twoRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES4_", [("a", "const $key&", "Ref{$key}"), ("b", key, "Any")])]
        fns[3]["is_noexcept"] = true                     # a noexcept function takes the thunk anyway
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "strparams"
        root = "$(escape_string(dir))"
        [link]
        enable_lto = false
        [wrap]
        language = "cpp"
        [types]
        strictness = "warn"
        allow_unknown_structs = true
        [cache]
        enabled = false
        """)
        cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
        lib = abspath(first(filter(p -> occursin("libjulia", basename(p)), Base.Libc.Libdl.dllist())))
        meta(r) = Dict{String,Any}("functions" => deepcopy(fns), "globals" => Dict{String,Any}(),
                                   "struct_definitions" => Dict{String,Any}(), "record_abi" => r)
        gen(r) = (res = _SW.generate_introspective_module_cpp(cfg, lib, meta(r), "StrParams",
                                                              _SW.create_type_registry(cfg), true);
                  res isa Tuple ? String(first(res)) : String(res))
        code = gen(ra)
        ex = Meta.parseall(code)
        @test !any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)
        chunk(f) = (m = match(Regex("function $f\\(.*?\\nend", "s"), code); m === nothing ? "" : m.match)
        # The spelling, in the loop and in the prediction: two methods of kind_of.
        @test occursin("function kind_of(s::$(_SW._STD_STRING_PARAM))", code)
        @test occursin("function kind_of(s::Cstring)", code) || occursin("function kind_of(s::Any)", code)
        @test length(collect(eachmatch(r"\nfunction kind_of\(", code))) == 2
        @test allunique(_SW._method_sig_keys(code))
        # The body: built before the call, its address the argument, ended in a finally.
        c = chunk("str_size_v")
        @test occursin("_rb_s1 = RepliBuild.StdStrings.string_arg(s, 32, LIBRARY_PATH, \"$c1\", true, \"$d1\", \"str_size_v\")", c)
        @test occursin("(try GC.@preserve _rb_s1 RepliBuild.JITManager.invoke(", c)
        @test occursin("RepliBuild.StdStrings.address(_rb_s1)) finally RepliBuild.StdStrings.release!(_rb_s1) end)", c)
        @test occursin("[Tier 2]", c)                     # noexcept, and still the thunk
        c2 = chunk("two")
        @test occursin("_rb_s1 = RepliBuild.StdStrings.string_arg(a, ", c2) && occursin("_rb_s2 = RepliBuild.StdStrings.string_arg(b, ", c2)
        @test occursin("GC.@preserve _rb_s1 _rb_s2 ", c2)
        @test occursin("finally RepliBuild.StdStrings.release!(_rb_s1); RepliBuild.StdStrings.release!(_rb_s2) end)", c2)
        # Old metadata: today's spelling, the reason beside the call.
        old = Dict{String,Any}(key => Dict{String,Any}("pass" => "reference", "byte_size" => "0x20"))
        code0 = gen(old)
        @test !occursin("string_arg", code0) && !occursin(_SW._STD_STRING_PARAM, code0)
        @test occursin("# std::string `$key` takes its byte stand-in: the metadata predates `string_layout`", code0)
        # Negative check: the plan neutered is the old wrapper, byte for byte.
        @eval _SW _string_param_plan(::AbstractString, record_abi) = nothing
        try
            res = Base.invokelatest(_SW.generate_introspective_module_cpp, cfg, lib, meta(ra), "StrParams",
                                    _SW.create_type_registry(cfg), true)
            code1 = res isa Tuple ? String(first(res)) : String(res)
            @test !occursin("string_arg", code1)
            @test occursin("function kind_of(s::Any)", code1) || occursin("function kind_of(s::Ref{", code1)
        finally
            # Restored from the file: the definition is longer than a line.
            Base.include_string(_SW, _plan_source())
        end
        @test Base.invokelatest(_SW._string_param_plan, "const $key&", ra) isa Tuple
    end
end

@testset "the destructor: the library's handle, then the C++ runtime by host" begin
    @test isempty(_SS.cxx_runtime_libraries(:sysv))
    @test isempty(_SS.cxx_runtime_libraries(:aapcs64))
    @test _SS.cxx_runtime_libraries(:win64) == ["libc++.dll", "libstdc++-6.dll"]
    # A symbol nothing defines: C_NULL, never an error.
    lib = Base.Libc.Libdl.dlpath(Base.Libc.Libdl.dlopen(Base.libllvm_path()))
    @test _SS.destructor(lib, "_ZN11no_such_type4noneD1Ev") == C_NULL
    # And with none, the stand-in comes back with a warning naming the function, once.
    slot = Ref(ntuple(i -> UInt8(i), 32))
    @test_logs (:warn, r"f_without_dtor: std::string returned as its raw bytes") begin
        @test _SS.take_string!(slot, _SS.string_reader(_LIBSTDCXX), lib, "_ZN11no_such_type4noneD1Ev",
                               "f_without_dtor") == slot[]
    end
    @test_logs _SS.take_string!(slot, _SS.string_reader(_LIBSTDCXX), lib, "_ZN11no_such_type4noneD1Ev",
                                "f_without_dtor")
end

@testset "the build records the layout, through bases and unions" begin
    # libstdc++'s string as `parse_dwarf_dump`'s toolchain table holds it.
    m(name, ct, off, size) = Dict{String,Any}("name" => name, "c_type" => ct, "offset" => off, "size" => size)
    key = "basic_string<char, std::char_traits<char>, std::allocator<char> >"
    defs = Dict{String,Dict{String,Any}}(
        key => Dict{String,Any}("kind" => "class", "byte_size" => "0x20",
                                "template_params" => Any[Dict{String,Any}("kind" => "type", "name" => "_CharT", "type" => "char")],
                                "members" => Any[m("_M_dataplus", "_Alloc_hider", "0x0", 8),
                                                 m("_M_string_length", "size_t", "0x8", 8),
                                                 m("_anon1", "_anon_u", "0x10", 16)]),
        "_Alloc_hider" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
                                           "members" => Any[m("_M_p", "char*", "0x0", 8)],
                                           "base_classes" => Any[Dict{String,Any}("type" => "allocator<char>",
                                                                                  "offset" => 0, "virtual" => false)]),
        "_anon_u" => Dict{String,Any}("kind" => "union", "byte_size" => "0x10",
                                      "members" => Any[m("_M_local_buf", "char[16]", "0x0", 16),
                                                       m("_M_allocated_capacity", "size_t", "0x0", 8)]))
    # `allocator<char>` has no definition but one byte: an empty base.
    sizes = t -> t == "allocator<char>" ? 1 : nothing
    L = _SC._string_layout(key, defs, Dict{String,Any}(), "_ZN…D4Ev"; byte_size_of = sizes)
    @test !haskey(L, "unread")
    @test L["char_type"] == "char" && L["destructor"] == "_ZN…D4Ev"
    @test [(l["path"], l["offset"]) for l in L["leaves"]] ==
          [("_M_dataplus._M_p", 0), ("_M_string_length", 8), ("_anon1._M_local_buf", 16),
           ("_anon1._M_allocated_capacity", 16)]
    @test _SS.string_reader(L) isa _SS.Reader
    # A base with no definition and no size at all (clang emits an `extern
    # template` allocator as a declaration-only DIE under libstdc++ 13):
    # `_Alloc_hider`'s own member `_M_p` covers its whole 8 bytes, so the base
    # is empty whatever the DWARF says, and the layout reads.
    L2 = _SC._string_layout(key, defs, Dict{String,Any}(), "_ZN…D4Ev")
    @test !haskey(L2, "unread")
    @test [(l["path"], l["offset"]) for l in L2["leaves"]] == [(l["path"], l["offset"]) for l in L["leaves"]]
    @test _SS.string_reader(L2) isa _SS.Reader
    # `_Alloc_hider` itself declaration-only (what libstdc++ 13's DWARF holds;
    # the parser keeps no entry for a declaration): the walk leaves `_M_dataplus`
    # a leaf of 8 bytes, and the reader takes it for `_M_p`.
    homed = deepcopy(defs)
    delete!(homed, "_Alloc_hider")
    homed[key]["members"][1]["size"] = 0          # a declaration has no byte_size
    L4 = _SC._string_layout(key, homed, Dict{String,Any}(), "_ZN…D4Ev")
    @test !haskey(L4, "unread")
    @test [(l["path"], l["offset"], l["size"]) for l in L4["leaves"]][1] == ("_M_dataplus", 0, 0)
    r4 = _SS.string_reader(L4)
    @test r4 isa _SS.Reader
    @test (r4.kind, r4.ptr, r4.len) == (:pointer_length, 0, 8)
    # The members leave room for the base: it may carry data we cannot see,
    # so the refusal stays, and says why.
    roomy = deepcopy(defs)
    roomy["_Alloc_hider"]["byte_size"] = "0x10"
    L3 = _SC._string_layout(key, roomy, Dict{String,Any}(), nothing)
    @test occursin("base `allocator<char>` has no definition", L3["unread"])
    @test occursin("leave room", L3["unread"])
    @test isempty(L3["leaves"])
    # The coverage test itself: tiled, a gap, overlapping union members, no size.
    @test _SC._members_cover_record(defs["_Alloc_hider"])
    @test _SC._members_cover_record(defs["_anon_u"])
    @test !_SC._members_cover_record(roomy["_Alloc_hider"])
    @test !_SC._members_cover_record(Dict{String,Any}("members" => Any[m("x", "int", "0x0", 4)]))
    @test !_SC._members_cover_record(Dict{String,Any}("byte_size" => "0x8", "members" => Any[]))
end

end
