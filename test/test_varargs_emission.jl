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

# A dropped duplicate is sometimes correct (a D1/D2 destructor pair) and
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
    @test occursin("Unreachable now", msg)

    @test U._chunk_mangled_symbol(a) == "_ZN5ImGui8TreeNodeEPKcPKcz"
    @test U._chunk_mangled_symbol("no docstring here") == "<unknown symbol>"
    # Varargs chunks carry no "Mangled symbol:" line; the @ccall names it
    @test U._chunk_mangled_symbol("""ptr = @ccall LIBRARY_PATH.var"fmt_msg"(f::Cstring;)::Cstring""") == "fmt_msg"
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
