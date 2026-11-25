# =============================================================================
# A reference to a type the wrapper has only a placeholder for (2026-11-24)
#
# A smart pointer and a `std::function` are `struct T end` in a wrapper: no
# value of that struct exists. A reference parameter is typed `Ref{T}`, so it
# takes only a pointer already typed that way. What the library hands a caller
# did not pass (the type matrix, 2026-11-23):
#
#   * `Ref` of the bytes a smart pointer was returned as, the manual's way for
#     a returned container, was a `MethodError`;
#   * a `std::function*` the library returned came back `Ptr{Cvoid}` (its `(`
#     was read as a function pointer's) and was a `MethodError` at a
#     `const std::function&`.
#
# Now a reference to a smart pointer also takes a `Ref` of exactly the bytes
# one is returned as, and a pointer to a record named with a function type is
# typed. The position is NOT `::Any`: tried first, a `@cfunction` pointer
# passed where a `std::function&` is taken was read as the object, and the
# process died. An untyped pointer is still refused there.
#
# No compiler: the decisions, and the wrapper text for made-up metadata.
# test_placeholder_references_live.jl calls a library.
# =============================================================================

using Test
using RepliBuild

const _PLR_W = RepliBuild.Wrapper
const _PLR_J = RepliBuild.JITManager

@testset "A reference to a smart pointer takes the bytes one is returned as" begin

    @testset "which reference also takes a Ref of bytes" begin
        placeholders = Set(["shared_ptr_Node", "unique_ptr_Node_std_default_delete_Node", "function_int_int", "Engine"])
        sp = "shared_ptr<Node>"
        up = "unique_ptr<Node, std::default_delete<Node> >"
        bytes(ct, c; ra = Dict{String,Any}()) = _PLR_W._value_ref_bytes(ct, c, placeholders, ra)
        # A reference to a smart pointer: the size it is returned as.
        @test bytes("Ref{shared_ptr_Node}", "const $sp&") == 16
        @test bytes("Ref{shared_ptr_Node}", "$sp&") == 16
        @test bytes("Ref{shared_ptr_Node}", "$sp&&") == 16
        @test bytes("Ref{unique_ptr_Node_std_default_delete_Node}", "const $up&") == 8
        # The size is the one a by-value return of it has: the DWARF's, then the table's.
        @test bytes("Ref{shared_ptr_Node}", "const $sp&"; ra = Dict{String,Any}(sp => Dict{String,Any}("byte_size" => "0x18"))) == 24
        # A placeholder that is no container has no bytes form: a std::function, a class only declared.
        @test bytes("Ref{function_int_int}", "const function<int (int)>&") == 0
        @test bytes("Ref{Engine}", "Engine&") == 0
        # Not a placeholder (a record with a layout), not a reference, not a single name.
        @test bytes("Ref{Rec}", "const Rec&") == 0
        @test bytes("Ptr{shared_ptr_Node}", "$sp*") == 0
        @test bytes("shared_ptr_Node", sp) == 0
        @test bytes("Ref{Ptr{shared_ptr_Node}}", "$sp*&") == 0

        # The signature's spelling: the typed pointer, or a Ref of exactly those bytes.
        spell(ct, c; kw...) = _PLR_W._by_address_ctype(ct, c; kw...)
        @test spell("Ref{shared_ptr_Node}", "const $sp&"; value_ref_bytes = 16) ==
              "Union{Ref{shared_ptr_Node}, Base.RefValue{NTuple{16, UInt8}}}"
        @test spell("Ref{shared_ptr_Node}", "const $sp&") == "Ref{shared_ptr_Node}"       # asked nothing: what it was
        @test spell("Ref{Engine}", "Engine&") == "Ref{Engine}"
        @test spell("shared_ptr_Node", sp; value_ref_bytes = 16) == "Ptr{Cvoid}"          # by value: its own rule first
        # It stays itself as a relaxed kind and at a strict position: a composed type.
        u = "Union{Ref{shared_ptr_Node}, Base.RefValue{NTuple{16, UInt8}}}"
        @test _PLR_W._relaxed_param_kind(u, t -> _PLR_W._type_name_bound(t, placeholders)) == u
        @test _PLR_W._strict_param_spelling(u; bound = t -> _PLR_W._type_name_bound(t, placeholders)) == u
        # And it is a type that takes exactly those two things.
        U = Union{Ref{Int8}, Base.RefValue{NTuple{16, UInt8}}}
        @test Ref(ntuple(_ -> 0x00, 16)) isa U && Ptr{Int8}(C_NULL) isa U && Ref(Int8(1)) isa U
        @test !(Ref(ntuple(_ -> 0x00, 8)) isa U) && !(Ptr{Cvoid}(C_NULL) isa U) && !(ntuple(_ -> 0x00, 16) isa U)
    end

    @testset "a pointer to a record named with a function type is typed" begin
        names = Set(["function_int_int", "Engine"])
        typed(t) = _PLR_W._typed_pointer_return(t, names, _PLR_W._sanitize_cpp_type_name)
        @test typed("const function<int (int)>*") == "Ptr{function_int_int}"
        @test typed("function<int (int)>&") == "Ptr{function_int_int}"
        @test typed("function<int (int)>**") == "Ptr{Ptr{function_int_int}}"
        @test typed("Engine*") == "Ptr{Engine}"
        # A function pointer is the type's own `(`, and stays untyped.
        @test typed("int (*)(int)") === nothing
        @test typed("void (*)(function<int (int)>)") === nothing
        @test typed("function_ptr(int)*") === nothing
        # A name the module does not emit stays untyped.
        @test typed("const function<void (int)>*") === nothing
    end

    @testset "the conversion takes the bytes by their address" begin
        # What the body tells `pointer_arg` is the placeholder's pointer, and a
        # Ref of plain bytes passes there by the address of those bytes.
        bytes = ntuple(i -> UInt8(i), 16)
        r = Ref(bytes)
        a = _PLR_J.pointer_arg(Ptr{_PLR_J.Opaque}, r)
        @test a.ptr == Base.unsafe_convert(Ptr{Cvoid}, r) && a.keep === r
    end

    @testset "the wrapper" begin
        sp = "shared_ptr<Node>"
        fnt = "function<int (int)>"
        fn(name, mangled, ret, retj, ps) = Dict{String,Any}("name" => name, "mangled" => mangled, "demangled" => name,
            "class" => "", "is_method" => false, "is_vararg" => false, "is_noexcept" => false,
            "has_object_pointer" => nothing,
            "parameters" => Any[Dict{String,Any}("name" => n, "c_type" => c, "julia_type" => j) for (n, c, j) in ps],
            "return_type" => Dict{String,Any}("c_type" => ret, "julia_type" => retj))
        fns = Any[
            fn("sp_read", "_Z7sp_readRKSt10shared_ptrI4NodeE", "int", "Cint", [("p", "const $sp&", "Ref{$sp}")]),
            fn("sp_make", "_Z7sp_makei", sp, sp, [("v", "int", "Cint")]),
            fn("fn_apply_ref", "_Z12fn_apply_refRKSt8functionIFiiEEi", "int", "Cint",
               [("f", "const $fnt&", "Ref{$fnt}"), ("x", "int", "Cint")]),
            fn("fn_twice", "_Z8fn_twicev", "const $fnt*", "Ptr{Cvoid}", []),
            # A record with a layout: a value of it exists, and `Ref(value)` is how it is passed.
            fn("rec_ref", "_Z7rec_refRK3Rec", "int", "Cint", [("r", "const Rec&", "Ref{Rec}")]),
            # A class only declared, alone and in an overload set that differs on it.
            fn("detach", "_Z6detachR6Engine", "int", "Cint", [("e", "Engine&", "Ref{Engine}")]),
            fn("attach", "_Z6attachR6Engine", "int", "Cint", [("e", "Engine&", "Ref{Engine}")]),
            fn("attach", "_Z6attachP6Engine", "int", "Cint", [("e", "Engine*", "Ptr{Engine}")])]
        structs = Dict{String,Any}("Rec" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x8", "scope" => "",
            "members" => Any[Dict{String,Any}("name" => "a", "c_type" => "int", "julia_type" => "Cint", "offset" => "0x0", "size" => 4),
                             Dict{String,Any}("name" => "b", "c_type" => "int", "julia_type" => "Cint", "offset" => "0x4", "size" => 4)]))
        ra = Dict{String,Any}(sp => Dict{String,Any}("byte_size" => "0x10", "pass" => "reference", "empty" => false))
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "placeholders"
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
        gen() = (res = Base.invokelatest(_PLR_W.generate_introspective_module_cpp, cfg, lib,
                     Dict{String,Any}("functions" => deepcopy(fns), "globals" => Dict{String,Any}(),
                                      "struct_definitions" => deepcopy(structs), "record_abi" => deepcopy(ra)),
                     "Placeholders", _PLR_W.create_type_registry(cfg), true);
                 res isa Tuple ? String(first(res)) : String(res))
        chunks(code, f) = [m.match for m in eachmatch(Regex("function $f\\(.*?\\nend", "s"), code)]
        code = gen()
        ex = Meta.parseall(code)
        @test !any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)
        # The placeholders this module declares.
        for name in ("shared_ptr_Node", "function_int_int", "Engine")
            @test occursin("\nstruct $name end\n", code)
        end
        @test occursin("struct Rec\n", code)

        # A reference to a smart pointer: its typed pointer or a Ref of its
        # bytes, converted to the placeholder's pointer.
        c = only(chunks(code, "sp_read"))
        @test occursin("function sp_read(p::Union{Ref{shared_ptr_Node}, Base.RefValue{NTuple{16, UInt8}}})", c)
        @test occursin("RepliBuild.JITManager.pointer_arg(Ptr{shared_ptr_Node}, p)", c)
        # The value a by-value return gives is those bytes.
        @test occursin("return_slot(NTuple{16, UInt8})", only(chunks(code, "sp_make")))
        # A std::function* the library returns is typed, so it passes at the
        # reference, which is what it was: no untyped pointer is taken there.
        @test occursin("_Z8fn_twicev_thunk\", Ptr{function_int_int})", only(chunks(code, "fn_twice")))
        @test occursin("function fn_apply_ref(f::Ref{function_int_int}, x::Integer)", only(chunks(code, "fn_apply_ref")))
        # A class only declared, and a record with a layout: what they were.
        @test occursin("function detach(e::Ref{Engine})", only(chunks(code, "detach")))
        @test occursin("function rec_ref(r::Ref{Rec})", only(chunks(code, "rec_ref")))
        # `attach(Engine&)` beside `attach(Engine*)`: two methods, as before.
        at = chunks(code, "attach")
        @test length(at) == 2
        @test count(c -> occursin("_Z6attachR6Engine", c), at) == 1 && count(c -> occursin("_Z6attachP6Engine", c), at) == 1
        @test allunique(_PLR_W._method_sig_keys(code))

        # A function the wrapper calls plainly (`noexcept`, and `nounwind` in
        # the library) takes the same two things: its call names an address.
        plain = deepcopy(fns)
        plain[1]["is_noexcept"] = true; plain[1]["nounwind"] = true
        fns_saved = copy(fns); empty!(fns); append!(fns, plain)
        pc = try only(chunks(gen(), "sp_read")) finally empty!(fns); append!(fns, fns_saved) end
        @test occursin("function sp_read(p::Union{Ref{shared_ptr_Node}, Base.RefValue{NTuple{16, UInt8}}})", pc)
        if occursin("ccall(", pc)
            @test occursin("Cint, (Ptr{Cvoid},), p)", pc)
        else
            @info "sp_read stayed on the thunk path in this configuration; the plain-call spelling is not exercised here"
        end
    end
end
