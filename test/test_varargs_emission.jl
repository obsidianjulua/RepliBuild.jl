# Varargs ABI emission regression (2026-07-10 C-generator audit).
#
# A variadic C callee must be called through a variadic call site: on x86-64
# SysV the callee's va_start prologue gates its XMM spill on AL, and only a
# variadic foreigncall sets AL. The generator therefore emits the @ccall
# semicolon form — never a flat non-variadic ccall type tuple (float varargs
# through that form only worked when leftover AL happened to be nonzero).
#
# String-level + macro-expansion checks only; no toolchain required.
#
# Imports are declared here rather than inherited from runtests.jl so the file
# also runs standalone (`julia --project=. test/test_varargs_emission.jl`) —
# it previously errored with `UndefVarError: @testset` down that path.

using Test
using RepliBuild

@testset "Varargs @ccall emission" begin
    W = RepliBuild.Wrapper

    params = [
        Dict{String,Any}("name" => "fmt", "julia_type" => "Cstring"),
        Dict{String,Any}("name" => "n", "julia_type" => "Cint"),
        Dict{String,Any}("name" => "varargs...", "julia_type" => ""),
    ]
    ret = Dict{String,Any}("julia_type" => "Cint")
    overloads = [["Cdouble"], ["Cstring", "Cint"]]

    code, exports = W.generate_vararg_wrappers(
        "test_printf", "test_printf", "test_printf",
        params, ret, overloads, false, "test_printf(const char*, int, ...)", :c)

    @test exports == ["test_printf", "test_printf_Cdouble", "test_printf_Cstring_Cint"]

    # Base wrapper: zero varargs passed, but the call site is still variadic
    # (trailing `;`) because the callee is — AL must be set either way
    @test occursin(
        "@ccall LIBRARY_PATH.var\"test_printf\"(fmt::Cstring, n_c::Cint;)::Cint", code)

    # Typed overloads: per-arg vararg types after the semicolon
    @test occursin(
        "@ccall LIBRARY_PATH.var\"test_printf\"(fmt::Cstring, n_c::Cint; va_1::Cdouble)::Cint", code)
    @test occursin(
        "@ccall LIBRARY_PATH.var\"test_printf\"(fmt::Cstring, n_c::Cint; va_1::Cstring, va_2::Cint)::Cint", code)

    # The flat non-variadic tuple form must not come back
    @test !occursin("ccall((:test_printf, LIBRARY_PATH)", code)

    # Emitted code is syntactically valid Julia
    parsed = Meta.parseall(code)
    @test !any(ex -> ex isa Expr && ex.head in (:error, :incomplete), parsed.args)

    # ABI property: every emitted @ccall must expand to a VARIADIC call.
    # On Julia 1.12 @ccall expands to `ccall(..., Expr(:cconv, _, nreq), ...)`
    # where nreq > 0 marks a variadic call with that many required args; a
    # non-variadic call site has nreq == 0 — that was the bug. All three
    # wrappers here have two fixed args, so nreq must be exactly 2.
    nreqs = Int[]
    collect_cconv(x) = nothing
    function collect_cconv(ex::Expr)
        ex.head === :cconv && push!(nreqs, ex.args[2])
        foreach(collect_cconv, ex.args)
    end
    # macroexpand does not descend into Expr(:toplevel), so expand each form
    for form in parsed.args
        form isa Expr && collect_cconv(macroexpand(@__MODULE__, form; recursive=true))
    end
    @test nreqs == [2, 2, 2]   # base + two overloads, all variadic call sites
end

# C default argument promotion means a narrow type never occupies a variadic
# slot. Declaring one is a WIDTH mistake that produces a well-formed ccall and
# wrong output — no crash, nothing downstream can catch it — so it has to be
# refused here. See `_VARARG_PROMOTED_TO`.
@testset "Un-promotable variadic types are refused" begin
    W = RepliBuild.Wrapper

    # float → double, and every integer type of rank below int → int
    for (narrow, promoted) in ("Cfloat" => "Cdouble", "Float32" => "Cdouble",
                               "Cchar" => "Cint", "Cshort" => "Cint",
                               "Cuchar" => "Cint", "Cushort" => "Cint",
                               "Bool" => "Cint", "Int8" => "Cint", "UInt16" => "Cint")
        err = try
            W._validate_vararg_type("myfmt", narrow); nothing
        catch e; e end
        @test err isa ErrorException
        # The error must name the offending entry AND the type to use — a bare
        # "invalid type" would leave the author guessing at the promotion rule.
        @test occursin("wrap.varargs.myfmt", err.msg)
        @test occursin(promoted, err.msg)
    end

    # The promoted spellings, pointers and strings all stay legal
    for ok in ("Cint", "Cuint", "Cdouble", "Cstring", "Csize_t", "Clonglong",
               "Ptr{Cvoid}", "Ptr{MyStruct}", "Any")
        @test W._validate_vararg_type("myfmt", ok) === nothing
    end

    # ...and the refusal reaches the real emission path, not just the predicate
    params = [Dict{String,Any}("name" => "fmt", "julia_type" => "Cstring"),
              Dict{String,Any}("name" => "varargs...", "julia_type" => "")]
    ret = Dict{String,Any}("julia_type" => "Cvoid")
    @test_throws ErrorException W.generate_vararg_wrappers(
        "myfmt", "myfmt", "myfmt", params, ret, [["Cfloat"]], false, "myfmt(const char*, ...)", :c)
end

# The variadic tail used to be emitted with its DECLARED type in the signature,
# so `f_Cint(fmt, 42)` was a MethodError (42 is Int64) and `f_Cstring(fmt, s)`
# rejected a String outright — while the fixed params next to them accepted
# both. Widening the signature is safe because each overload is its own named
# function, and the @ccall must keep the declared type or the ABI moves.
@testset "Variadic tail accepts ergonomic argument types" begin
    W = RepliBuild.Wrapper

    @test W._vararg_sig_type("Cint")    == "Integer"
    @test W._vararg_sig_type("Csize_t") == "Integer"
    @test W._vararg_sig_type("Cdouble") == "Real"
    @test W._vararg_sig_type("Cstring") == "Union{AbstractString,Cstring}"
    @test W._vararg_sig_type("Ptr{Cvoid}") == "Any"
    @test W._vararg_sig_type("Any")     == "Any"

    params = [Dict{String,Any}("name" => "fmt", "julia_type" => "Cstring"),
              Dict{String,Any}("name" => "varargs...", "julia_type" => "")]
    ret = Dict{String,Any}("julia_type" => "Cvoid")
    code, _ = W.generate_vararg_wrappers(
        "myfmt", "myfmt", "myfmt", params, ret,
        [["Cint"], ["Cdouble"], ["Cstring", "Cint"]], false,
        "myfmt(const char*, ...)", :c)

    # Signature widened...
    @test occursin("function myfmt_Cint(fmt::Cstring, va_1::Integer)", code)
    @test occursin("function myfmt_Cdouble(fmt::Cstring, va_1::Real)", code)
    @test occursin("va_1::Union{AbstractString,Cstring}, va_2::Integer", code)

    # ...while every @ccall keeps the DECLARED type. This is the ABI invariant:
    # if these move, the variadic slots change width.
    @test occursin("; va_1::Cint)::Cvoid", code)
    @test occursin("; va_1::Cdouble)::Cvoid", code)
    @test occursin("; va_1::Cstring, va_2::Cint)::Cvoid", code)

    # Overload NAMES still come from the declared types, not the widened ones
    @test occursin("function myfmt_Cstring_Cint(", code)
    @test !occursin("myfmt_Integer", code)

    parsed = Meta.parseall(code)
    @test !any(ex -> ex isa Expr && ex.head in (:error, :incomplete), parsed.args)
end

# A dropped duplicate is sometimes correct (a D0/D1/D2 destructor variant) and
# sometimes a distinct C++ entry point going unreachable (an ::Any-collapsed
# overload). The count alone cannot tell those apart, so the message names the
# symbols.
@testset "Dedup names the symbols it drops" begin
    U = RepliBuild.Wrapper

    mk(sym, body) = """
        \"\"\"
        # Metadata
        - Mangled symbol: `$sym`
        \"\"\"
        $body
        """
    # Same Julia signature, two different C++ symbols — the TreeNode shape
    a = mk("_ZN5ImGui8TreeNodeEPKcPKcz", "function ImGui_TreeNode(a::Any, b::Any)\n    nothing\nend")
    b = mk("_ZN5ImGui8TreeNodeEPKvPKcz", "function ImGui_TreeNode(a::Any, b::Any)\n    nothing\nend")

    logs, kept = Test.collect_test_logs() do
        U._dedup_method_chunks([a, b])
    end
    @test length(kept) == 1
    @test occursin("PKvPKcz", kept[1])   # last definition kept

    msg = join([string(r.message) for r in logs], "\n")
    @test occursin("_ZN5ImGui8TreeNodeEPKcPKcz", msg)   # the one that vanished
    @test occursin("_ZN5ImGui8TreeNodeEPKvPKcz", msg)   # what shadowed it
    @test occursin("UNREACHABLE", msg)
    # TreeNode must be classified as SIGNAL, not filed under the benign bucket:
    # two genuinely different C++ entry points collapsed by ::Any widening.
    @test occursin("1 distinct C++ entry point(s) collapsed", msg)
    @test !occursin("destructor/alias variant", msg)

    @test U._chunk_mangled_symbol(a) == "_ZN5ImGui8TreeNodeEPKcPKcz"
    @test U._chunk_mangled_symbol("no docstring here") == "<unknown symbol>"
    # Varargs chunks carry no "Mangled symbol:" line; the @ccall names it
    @test U._chunk_mangled_symbol("""ptr = @ccall LIBRARY_PATH.var"fmt_msg"(f::Cstring;)::Cstring""") == "fmt_msg"

    # The other half of the partition. D0/D1/D2 are the deleting /
    # complete-object / base-object variants of ONE destructor: distinct symbols,
    # one Julia method by construction. llamacpp drops 210 of these against 24
    # real collapses, so failing to bucket them buries the 24 that matter.
    @test U._dtor_variant_group("_ZN4ggml3cpu17extra_buffer_typeD2Ev") == "_ZN4ggml3cpu17extra_buffer_type"
    @test U._dtor_variant_group("_ZN4ggml3cpu17extra_buffer_typeD0Ev") ==
          U._dtor_variant_group("_ZN4ggml3cpu17extra_buffer_typeD2Ev")
    @test U._dtor_variant_group("_ZN4FooD2Ev") != U._dtor_variant_group("_ZN4BarD0Ev")
    # A real overload must never be grouped away as a destructor
    @test U._dtor_variant_group("_ZN17llm_graph_context10build_attnEP22llm_graph_input_attn_kP11ggml_tensorfi") === nothing

    d0 = mk("_ZN9time_measD0Ev", "function time_meas_destroy(a::Any)\n    nothing\nend")
    d2 = mk("_ZN9time_measD2Ev", "function time_meas_destroy(a::Any)\n    nothing\nend")
    dlogs, dkept = Test.collect_test_logs() do
        U._dedup_method_chunks([d0, d2])
    end
    @test length(dkept) == 1
    dmsg = join([string(r.message) for r in dlogs], "\n")
    @test occursin("1 destructor/alias variant(s)", dmsg)
    # …and it must NOT be reported as a lost entry point
    @test !occursin("distinct C++ entry point(s) collapsed", dmsg)
end

# A parameter TYPE may contain commas. Splitting the argument list on every
# comma turns one argument into two and changes the dispatch key that dedup
# decides on — so two different signatures can collide, or one chunk can be
# dropped against a key nothing really claimed. Latent until a generated
# signature used a comma-bearing type; `Union{AbstractString,Cstring}` does.
@testset "Signature keys split on top-level commas only" begin
    U = RepliBuild.Wrapper

    @test U._split_toplevel_commas("a::Int, b::Float64") |> length == 2
    @test U._split_toplevel_commas("a::Union{AbstractString,Cstring}") |> length == 1
    @test U._split_toplevel_commas("a::NTuple{8,UInt8}, b::Int") |> length == 2
    @test U._split_toplevel_commas("a::Dict{Symbol,Vector{Int}}, b::Ptr{Cvoid}") |> length == 2
    @test U._split_toplevel_commas("") |> length == 1
    # unbalanced closer must not drive depth negative and re-enable splitting
    @test U._split_toplevel_commas("a::T}, b::U") |> length == 2

    # The key itself: a comma-bearing type stays ONE argument
    k = U._method_sig_keys("function f(a::Union{AbstractString,Cstring}, b::Integer)\nend")
    @test k == ["f(Union{AbstractString,Cstring},Integer)"]

    # And the arity it implies is right — a 2-arg and a 3-arg method whose
    # types differ only inside braces must NOT collide
    k2 = U._method_sig_keys("function g(a::Union{A,B})\nend\nfunction g(a::A, b::B)\nend")
    @test k2[1] != k2[2]
end

# The VARARGS INTERCEPTION branch `continue`d before every guard the plain
# path applies to a by-value record but the conflict trap, so a
# `[wrap.varargs]` function whose fixed parameters or return held a packed or
# float-bearing record of 16 bytes or less, a `long double`, or an over-aligned
# MEMORY-class record got a plain @ccall with the record spelled by name
# (review W5, 2026-10-24). Both paths now take one verdict
# (`GeneratorC._c_byvalue_verdict`): the same trap stubs, BRANCH 0's explicit
# sret, and the byval carrier. Generated for both conventions on any host.
@testset "Varargs fixed parameters get the plain path's guards (W5)" begin
    using Libdl
    W = RepliBuild.Wrapper
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "w5"
    root = "$(escape_string(dir))"
    [link]
    enable_lto = false
    [wrap]
    language = "c"
    [wrap.varargs]
    va_packed = [["Cint"]]
    va_make_packed = [["Cint"]]
    va_dbits = [["Cint"]]
    va_d4a = [["Cint"]]
    va_a32 = [["Cint"]]
    va_ld = [["Cint"]]
    va_make_d4a = [["Cint"]]
    va_plain = [["Cint"]]
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
    M(n, jt, ct, off, sz) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct, "offset" => off, "size" => sz)
    structs = Dict{String,Any}(
        # 6 bytes, `int` at offset 1: SysV MEMORY, its blob INTEGER.
        "PackedInt" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x6",
                                        "members" => Any[M("tag", "Cchar", "char", "0x0", 1), M("val", "Cint", "int", "0x1", 4)]),
        # 16 bytes, a double and a bitfield: a float blob, SSE on SysV.
        "DBits" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10",
                                    "members" => Any[M("d", "Cdouble", "double", "0x0", 8),
                                                     Dict{String,Any}("name" => "flags", "julia_type" => "Cuint", "c_type" => "unsigned int",
                                                                      "offset" => nothing, "size" => 4, "bit_size" => 3, "data_bit_offset" => 64)]),
        "Big24" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x18",
                                    "members" => Any[M("a", "Clonglong", "long long", "0x0", 8), M("b", "Clonglong", "long long", "0x8", 8),
                                                     M("c", "Cint", "int", "0x10", 4)]),
        "D4A" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x20", "alignment" => 16,
                                  "members" => Any[M("a", "Cdouble", "double", "0x0", 8), M("b", "Cdouble", "double", "0x8", 8),
                                                   M("c", "Cdouble", "double", "0x10", 8), M("d", "Cdouble", "double", "0x18", 8)]),
        "A32" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x20", "alignment" => 32,
                                  "members" => Any[M("d", "NTuple{2, Cdouble}", "double[2]", "0x0", 16)]))
    P(n, t) = Dict{String,Any}("name" => n, "julia_type" => t, "c_type" => t)
    VA = Dict{String,Any}("name" => "varargs...", "julia_type" => "", "c_type" => "...")
    F(name, ps, ret; facts = nothing) = begin
        f = Dict{String,Any}("name" => name, "mangled" => name, "demangled" => name, "is_vararg" => true,
                             "parameters" => Any[ps..., VA], "return_type" => P("", ret))
        facts === nothing || (f["byval_params"] = Any[Dict{String,Any}("record" => r, "align" => a) for (r, a) in facts])
        f
    end
    fmt = P("fmt", "Cstring")
    fns = Any[F("va_packed", [P("v", "PackedInt"), fmt], "PackedInt"),
              F("va_make_packed", [fmt], "PackedInt"),
              F("va_dbits", [P("v", "DBits"), fmt], "Cvoid"),
              F("va_d4a", [P("p", "Big24"), P("v", "D4A"), fmt], "Cdouble"; facts = [("Big24", 8), ("D4A", 16)]),
              F("va_a32", [P("p", "Big24"), P("v", "A32"), fmt], "Cdouble"; facts = [("Big24", 8), ("A32", 32)]),
              F("va_ld", [Dict{String,Any}("name" => "x", "julia_type" => "Any", "c_type" => "long double"), fmt], "Cvoid"),
              F("va_make_d4a", [fmt], "D4A"),
              F("va_plain", [P("p", "Big24"), fmt], "Cint"; facts = [("Big24", 8)])]
    meta = Dict{String,Any}("functions" => fns, "struct_definitions" => structs, "globals" => Dict{String,Any}())
    libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    gen(abi) = W.generate_introspective_module_c(cfg, libref, meta, "W5", W.create_type_registry(cfg), false; abi)
    body(code, name) = let i = findfirst("function $name(", code)
        i === nothing ? "" : (r = SubString(code, first(i)); j = findfirst(r"\nend\b", r);
                              j === nothing ? String(r) : String(r[1:last(j)]))
    end
    form(code, name) = let b = body(code, name)
        isempty(b) ? :missing :
        occursin("ABI Safety Trap", b) ? :trap :
        occursin("ret_ptr", b) ? :aligned :
        occursin("ret_buf", b) ? :sret :
        occursin("_rb_byval16(", b) ? :carrier :
        occursin("@ccall LIBRARY_PATH.var\"$(replace(name, r"_Cint$" => ""))\"", b) ? :ccall : :other
    end
    sysv, win64 = gen(:sysv), gen(:win64)
    # Both the base wrapper and each overload, on both conventions.
    #          function           SysV      Win64
    table = [("va_packed",       :trap,    :ccall),   # takes PackedInt: MEMORY vs blob INTEGER
             ("va_packed_Cint",  :trap,    :ccall),
             ("va_make_packed",  :sret,    :ccall),   # returns it: the explicit sret; Win64 RAX by size
             ("va_make_packed_Cint", :sret, :ccall),
             ("va_dbits",        :trap,    :ccall),   # a float blob in the register window
             ("va_dbits_Cint",   :trap,    :ccall),
             ("va_d4a",          :carrier, :carrier), # the 16-aligned slot after Big24 (W2); Win64 the 16-aligned copy
             ("va_d4a_Cint",     :carrier, :carrier),
             ("va_a32",          :trap,    :carrier), # 32-aligned: Julia cannot place a SysV slot; Win64 gives the copy 16
             ("va_a32_Cint",     :trap,    :carrier),
             ("va_ld",           :trap,    :trap),    # long double, on both
             ("va_ld_Cint",      :trap,    :trap),
             ("va_make_d4a",     :aligned, :aligned), # returns a 16-aligned record: the wrapper's own buffer
             ("va_make_d4a_Cint", :aligned, :aligned),
             ("va_plain",        :ccall,   :ccall),   # an 8-aligned MEMORY record: Julia's own byval
             ("va_plain_Cint",   :ccall,   :ccall)]
    checked = 0
    for (name, want_sysv, want_win64) in table
        @test (name, :sysv, form(sysv, name)) == (name, :sysv, want_sysv)
        @test (name, :win64, form(win64, name)) == (name, :win64, want_win64)
        checked += 1
    end
    @test checked == length(table)
    # What each shape says.
    @test occursin("takes `PackedInt` by value", body(sysv, "va_packed_Cint"))
    @test occursin("SysV register window", body(sysv, "va_packed"))
    @test occursin("takes `DBits` by value", body(sysv, "va_dbits_Cint"))
    @test occursin("long double", body(sysv, "va_ld_Cint")) && occursin("long double", body(win64, "va_ld"))
    @test occursin("takes `A32` by value in a 32-byte-aligned stack slot", body(sysv, "va_a32_Cint"))
    # The explicit sret: the buffer first, then the fixed arguments, then the tail.
    @test occursin("ret_buf = Ref(PackedInt())", body(sysv, "va_make_packed"))
    @test occursin("@ccall LIBRARY_PATH.var\"va_make_packed\"(ret_buf::Ptr{PackedInt}, fmt::Cstring;)::Cvoid", body(sysv, "va_make_packed"))
    @test occursin("@ccall LIBRARY_PATH.var\"va_make_packed\"(ret_buf::Ptr{PackedInt}, fmt::Cstring; va_1::Cint)::Cvoid", body(sysv, "va_make_packed_Cint"))
    @test occursin("return ret_buf[]", body(sysv, "va_make_packed_Cint"))
    @test occursin("function va_make_packed_Cint(fmt::Cstring, va_1::Integer)::PackedInt", body(sysv, "va_make_packed_Cint"))
    @test occursin("@ccall LIBRARY_PATH.var\"va_make_packed\"(fmt::Cstring; va_1::Cint)::PackedInt", body(win64, "va_make_packed_Cint"))
    # The aligned buffer, as BRANCH 0 shapes it.
    @test occursin("ret_buf = Ref{NTuple{47, UInt8}}()", body(sysv, "va_make_d4a_Cint"))
    @test occursin("ret_ptr = Ptr{D4A}((UInt(Base.unsafe_convert(Ptr{Cvoid}, ret_buf)) + 15) & ~UInt(15))", body(sysv, "va_make_d4a_Cint"))
    @test occursin("@ccall LIBRARY_PATH.var\"va_make_d4a\"(ret_ptr::Ptr{D4A}, fmt::Cstring; va_1::Cint)::Cvoid", body(sysv, "va_make_d4a_Cint"))
    @test occursin("return unsafe_load(ret_ptr)", body(win64, "va_make_d4a"))
    # The carrier: the record in the signature, its bytes in the @ccall.
    @test occursin("function va_d4a_Cint(p::Big24, v::D4A, fmt::Cstring, va_1::Integer)::Cdouble", body(sysv, "va_d4a_Cint"))
    @test occursin("v_c = _rb_byval16(v, _RB_ByVal16{2})", body(sysv, "va_d4a_Cint"))
    @test occursin("@ccall LIBRARY_PATH.var\"va_d4a\"(p::Big24, v_c::_RB_ByVal16{2}, fmt::Cstring; va_1::Cint)::Cdouble", body(sysv, "va_d4a_Cint"))
    @test occursin("@ccall LIBRARY_PATH.var\"va_d4a\"(p::Big24, v_c::_RB_ByVal16{2}, fmt::Cstring;)::Cdouble", body(sysv, "va_d4a"))
    @test occursin("@ccall LIBRARY_PATH.var\"va_d4a\"(p::Big24, v_c::_RB_ByVal16{2}, fmt::Cstring; va_1::Cint)::Cdouble", body(win64, "va_d4a_Cint"))
    @test count("struct _RB_ByVal16{N}", sysv) == 1 && count("struct _RB_ByVal16{N}", win64) == 1
    @test occursin("@ccall LIBRARY_PATH.var\"va_plain\"(p::Big24, fmt::Cstring; va_1::Cint)::Cint", body(sysv, "va_plain_Cint"))
    # Every variadic call site is still variadic, sret and carrier included,
    # and both modules parse.
    for (code, abi) in ((sysv, :sysv), (win64, :win64))
        ex = Meta.parseall(code)
        @test (abi, any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)) == (abi, false)
        nreqs = Int[]
        collect_cconv(x) = nothing
        function collect_cconv(ex::Expr)
            ex.head === :cconv && push!(nreqs, ex.args[2])
            foreach(collect_cconv, ex.args)
        end
        for f in ("va_make_packed_Cint", "va_d4a_Cint", "va_plain_Cint")
            b = body(code, f)
            occursin("@ccall", b) || continue
            for form_ in Meta.parseall(b).args
                form_ isa Expr && collect_cconv(macroexpand(@__MODULE__, form_; recursive = true))
            end
        end
        @test (abi, all(n -> n > 0, nreqs), length(nreqs) >= 2) == (abi, true, true)
    end
end
