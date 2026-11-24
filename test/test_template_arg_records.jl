# =============================================================================
# Records named with a pointer, array or function type in their template
# arguments, and the size of a container crossed by value (2026-11-23)
#
# Found by the type matrix, two defects with one shape: a fact about a record
# was read from its NAME where the DWARF had the answer.
#
#   * `_by_value_record_name` read `*`, `&`, `(` and `[` over the whole
#     spelling, so `unique_ptr<int[], default_delete<int[]>>`, `vector<Node *>`,
#     `function<int (int)>` and `Tag<int *>` had no `record_abi` entry. Nothing
#     then said that C++ returns the record through a hidden pointer, or that it
#     is empty: `std::unique_ptr<int[]>` was returned as eight bytes in a
#     register while the callee wrote through the first argument register.
#   * A container's by-value size came from a table keyed on the template's
#     name. `std::vector<bool>` is 40 bytes in libstdc++, not a vector's 24: the
#     wrapper's return buffer was 24 bytes and the callee built 40 in it.
#
# No compiler: the name, the size, the crossing verdict under each convention,
# and the wrapper text for made-up metadata. test_template_arg_records_live.jl
# calls a library.
# =============================================================================

using Test
using RepliBuild

const _TAR_C = RepliBuild.Compiler
const _TAR_FG = RepliBuild.JLCSIRGenerator.FunctionGen
const _TAR_W = RepliBuild.Wrapper

const _TAR_VB = "vector<bool, std::allocator<bool> >"
const _TAR_VI = "vector<int, std::allocator<int> >"
const _TAR_UPA = "unique_ptr<int[], std::default_delete<int[]> >"
const _TAR_FN = "function<int (int)>"

@testset "Records named with a declarator in their template arguments" begin

    @testset "a declarator inside the template arguments is not the type's" begin
        name = _TAR_C._by_value_record_name
        # Records: the spelling is the key the record's facts are filed under.
        for t in (_TAR_UPA, _TAR_FN, "vector<Node *, std::allocator<Node *> >", "Tag<int *>",
                  "fstring<const char *>", "pair<int &, int>", "Holder<int (*)(int)>",
                  "map<int, char[4], std::less<int>, std::allocator<std::pair<const int, char[4]> > >")
            @test name(t) == t
        end
        @test name("const Tag<int *>") == "Tag<int *>"            # only top-level cv goes
        @test name("Tag<const int *> const") == "Tag<const int *>"
        # Not records: the type's own declarator, whatever its arguments hold.
        for t in ("Tag<int *> *", "Tag<int *>&", "Tag<int *> &&", "Tag<int>[2]", "int[4]", "int *",
                  "function_ptr(int)", "void (*)(Tag<int *>)", "void", "unknown", "")
            @test name(t) === nothing
        end
    end

    @testset "a container's by-value size is the DWARF's, then the table's" begin
        bytes = _TAR_FG.stl_value_bytes
        ra = Dict{String,Any}(
            _TAR_VB => Dict{String,Any}("byte_size" => "0x28", "pass" => "reference", "empty" => false),
            _TAR_VI => Dict{String,Any}("byte_size" => "0x18", "pass" => "reference", "empty" => false),
            # Another C++ library's string: 24 bytes where the table says 32.
            "basic_string<char, std::char_traits<char>, std::allocator<char> >" =>
                Dict{String,Any}("byte_size" => 24, "pass" => "reference"),
            "map<int, int>" => Dict{String,Any}("byte_size" => "0x30", "conflict" => true))
        @test bytes(_TAR_VB, ra) == 40
        @test bytes(_TAR_VI, ra) == 24
        @test bytes("const " * _TAR_VB, ra) == 40                                  # top-level cv
        @test bytes("std::" * _TAR_VB, ra) == 40                                   # a scoped spelling finds the bare key
        @test bytes("basic_string<char, std::char_traits<char>, std::allocator<char> >", ra) == 24
        # No fact for this instantiation: the table.
        @test bytes("vector<double, std::allocator<double> >", ra) == 24
        @test bytes("deque<int, std::allocator<int> >", Dict{String,Any}()) == 80
        # Facts that stand for two types say nothing.
        @test bytes("map<int, int>", ra) == 48
        # Not a container: no size, whatever the facts say.
        @test bytes("pair<int, int>", Dict{String,Any}("pair<int, int>" => Dict{String,Any}("byte_size" => "0x8"))) == 0
        @test bytes("int", ra) == 0

        # The table must never be smaller than the object it stands for:
        # `vector<bool>` is the bit-packed specialization.
        size = RepliBuild.stl_container_size
        for t in (_TAR_VB, "std::vector<bool>", "std::vector<bool, std::allocator<bool> >", "const vector<bool>")
            @test size(t) == 40
        end
        for t in (_TAR_VI, "std::vector<double>", "vector<boolean_t>", "vector<bool *>",
                  "vector<std::vector<bool, std::allocator<bool> >, std::allocator<std::vector<bool, std::allocator<bool> > > >")
            @test size(t) == 24
        end
        @test bytes(_TAR_VB, Dict{String,Any}()) == 40
    end

    @testset "the crossing, under each convention" begin
        structs = Dict{String,Any}()
        func(ret) = Dict{String,Any}("return_type" => Dict{String,Any}("c_type" => ret), "parameters" => Any[])
        byref(n) = Dict{String,Any}("byte_size" => n, "pass" => "reference", "empty" => false)
        with = Dict{String,Any}(_TAR_UPA => byref("0x8"), _TAR_VB => byref("0x28"), _TAR_FN => byref("0x20"),
                                "Tag<int *>" => Dict{String,Any}("byte_size" => "0x1", "pass" => "value", "empty" => true))
        without = Dict{String,Any}()
        checked = 0
        for abi in (:sysv, :win64, :aapcs64)
            # A container C++ returns through a hidden pointer is built in the
            # caller's buffer; with no fact (metadata from before the fix: a
            # rebuild) it is not.
            for t in (_TAR_UPA, _TAR_VB)
                @test _TAR_FG.returns_in_caller_buffer(func(t), structs, with; abi)
                @test !_TAR_FG.returns_in_caller_buffer(func(t), structs, without; abi)
            end
            # A record with no layout that C++ passes by reference crosses by
            # address; with no fact it is untypable.
            @test _TAR_FG.by_value_crossing(_TAR_FN, structs, with; abi) == (:indirect, nothing)
            @test _TAR_FG.by_value_crossing(_TAR_FN, structs, without; abi) == (:opaque, nothing)
            checked += 1
        end
        @test checked == 3
        # An empty record is what each convention says an empty record is:
        # dropped, a register pad, a register pad in and nothing out.
        tag(abi; ret = false) = _TAR_FG.by_value_crossing("Tag<int *>", structs, with; abi, ret)
        @test tag(:sysv) == (:ignore, nothing)
        @test tag(:win64) == (:regpad, "i8")
        @test tag(:aapcs64) == (:regpad, "i64")
        @test tag(:aapcs64; ret = true) == (:ignore, nothing)

        # A by-value parameter the thunk takes as the address of the caller's
        # object: C++ passes it by reference and nothing here lays it out.
        addr = _TAR_FG.takes_object_address
        @test addr(_TAR_FN, structs, with)
        @test addr("const " * _TAR_FN, structs, with)
        @test !addr(_TAR_FN, structs, without)                       # no fact: untypable, a refusal
        @test !addr(_TAR_VB, structs, with)                          # a container: its own conversion
        @test !addr("Tag<int *>", structs, with)                     # an empty record: nothing is passed
        @test !addr("int", structs, with)
        laid = Dict{String,Any}("Holder<int *>" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10",
                                                                    "members" => Any[]))
        @test !addr("Holder<int *>", laid, Dict{String,Any}("Holder<int *>" => byref("0x10")))   # it has a layout: its bytes cross
    end

    @testset "the wrapper: one size for the buffer, the value and the parameter" begin
        fn(name, mangled, ret, ps) = Dict{String,Any}("name" => name, "mangled" => mangled, "demangled" => name,
            "class" => "", "is_method" => false, "is_vararg" => false, "is_noexcept" => false,
            "has_object_pointer" => nothing,
            "parameters" => Any[Dict{String,Any}("name" => n, "c_type" => c, "julia_type" => j) for (n, c, j) in ps],
            "return_type" => Dict{String,Any}("c_type" => ret, "julia_type" => ret == "int" ? "Cint" : ret))
        fns = Any[fn("vb_make", "_Z7vb_makei", _TAR_VB, [("n", "int", "Cint")]),
                  fn("vb_count_val", "_Z12vb_count_valSt6vectorIbSaIbEE", "int", [("v", _TAR_VB, "Any")]),
                  fn("vi_make", "_Z7vi_makei", _TAR_VI, [("n", "int", "Cint")]),
                  fn("upa_make", "_Z8upa_makei", _TAR_UPA, [("n", "int", "Cint")]),
                  fn("fn_apply", "_Z8fn_applySt8functionIFiiEEi", "int", [("f", _TAR_FN, _TAR_FN), ("x", "int", "Cint")]),
                  # A reference to the same record: the name is bound to an empty struct.
                  fn("fn_empty", "_Z8fn_emptyRKSt8functionIFiiEE", "int", [("f", "const $(_TAR_FN)&", "Ref{$(_TAR_FN)}")])]
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "tmplargs"
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
        gen(ra) = (res = _TAR_W.generate_introspective_module_cpp(cfg, lib,
                       Dict{String,Any}("functions" => deepcopy(fns), "globals" => Dict{String,Any}(),
                                        "struct_definitions" => Dict{String,Any}(), "record_abi" => ra),
                       "TmplArgs", _TAR_W.create_type_registry(cfg), true);
                   res isa Tuple ? String(first(res)) : String(res))
        chunk(code, f) = (m = match(Regex("function $f\\(.*?\\nend", "s"), code); m === nothing ? "" : m.match)
        byref(n) = Dict{String,Any}("byte_size" => n, "pass" => "reference", "empty" => false)

        code = gen(Dict{String,Any}(_TAR_VB => byref("0x28"), _TAR_VI => byref("0x18"), _TAR_UPA => byref("0x8"),
                                    _TAR_FN => byref("0x20")))
        ex = Meta.parseall(code)
        @test !any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)
        # The buffer the callee builds a vector<bool> in is as large as one.
        c = chunk(code, "vb_make")
        @test occursin("return_slot(NTuple{40, UInt8})", c)
        @test occursin("_Z7vb_makei_thunk_inplace", c)
        @test !occursin("NTuple{24, UInt8}", c)
        # And a by-value parameter takes the value that return gives.
        @test occursin("ByValue{NTuple{40, UInt8}}", chunk(code, "vb_count_val"))
        # Every other vector is what it was.
        @test occursin("return_slot(NTuple{24, UInt8})", chunk(code, "vi_make"))
        # unique_ptr<int[]>, with its fact: in the caller's buffer.
        c = chunk(code, "upa_make")
        @test occursin("return_slot(NTuple{8, UInt8})", c)
        @test occursin("_Z8upa_makei_thunk_inplace", c)
        # The wrapper and the thunk generator are asked the same question.
        @test _TAR_FG.returns_in_caller_buffer(fns[4], Dict{String,Any}(), Dict{String,Any}(_TAR_UPA => byref("0x8")))

        # std::function by value: the thunk takes the address of the caller's
        # object, so the argument is converted like any pointer, and the
        # position is `::Any`. Typed by the record's own name it took only a
        # value of the empty struct that name is bound to; passed as it came,
        # a Julia function was boxed and the box read as the object.
        c = chunk(code, "fn_apply")
        @test occursin("function fn_apply(f::Any, x::Integer)", c)
        @test occursin("RepliBuild.JITManager.pointer_arg(Ptr{RepliBuild.JITManager.Opaque}, f), Cint(x))", c)
        @test !occursin("ABI Safety Trap", c)
        @test occursin("function fn_empty(f::Ref{function_int_int})", code)       # a reference is what it was
        @test allunique(_TAR_W._method_sig_keys(code))

        # A `std::vector<bool>` listed in `[types] templates` gets no factory:
        # it is not a vector of `bool`s a CppVector could read. Every other
        # vector keeps its own.
        @test _TAR_W._is_bool_vector_element("bool")
        @test _TAR_W._is_bool_vector_element("bool, std::allocator<bool> ")
        @test !_TAR_W._is_bool_vector_element("boolean_t")
        @test !_TAR_W._is_bool_vector_element("bool *")
        @test !_TAR_W._is_bool_vector_element("std::vector<bool>")
        row(c, m, sym) = Dict{String,Any}("method" => m, "mangled" => sym, "container" => c)
        stl = Dict{String,Any}(
            "std::vector<bool>" => Any[row("std::vector<bool>", "size", "_ZNKSt6vectorIbSaIbEE4sizeEv")],
            "std::vector<int>" => Any[row("std::vector<int>", "size", "_ZNKSt6vectorIiSaIiEE4sizeEv")])
        res = @test_logs (:warn, r"no CppVector factory for \"std::vector<bool>\".*bit-packed") match_mode = :any begin
            _TAR_W.generate_introspective_module_cpp(cfg, lib,
                Dict{String,Any}("functions" => deepcopy(fns), "globals" => Dict{String,Any}(),
                                 "struct_definitions" => Dict{String,Any}(), "record_abi" => Dict{String,Any}(),
                                 "stl_methods" => stl),
                "TmplArgs", _TAR_W.create_type_registry(cfg), true)
        end
        fcode = res isa Tuple ? String(first(res)) : String(res)
        @test occursin("function create_std_vector_int()", fcode)
        @test !occursin("create_std_vector_bool", fcode)

        # Metadata with no facts (built before the fix): the table, which
        # is never smaller than the object, and the old thunk's name. A
        # std::function by value is the refusal it was.
        code0 = gen(Dict{String,Any}())
        @test occursin("ABI Safety Trap: cannot call 'fn_apply'", chunk(code0, "fn_apply"))
        c0 = chunk(code0, "vb_make")
        @test occursin("NTuple{40, UInt8}", c0) && !occursin("_thunk_inplace", c0)
        @test occursin("NTuple{8, UInt8}", chunk(code0, "upa_make")) && !occursin("_thunk_inplace", chunk(code0, "upa_make"))
    end
end
