# Varargs ABI emission regression (2026-07-10 C-generator audit).
#
# A variadic C callee must be called through a variadic call site: on x86-64
# SysV the callee's va_start prologue gates its XMM spill on AL, and only a
# variadic foreigncall sets AL. The generator therefore emits the @ccall
# semicolon form — never a flat non-variadic ccall type tuple (float varargs
# through that form only worked when leftover AL happened to be nonzero).
#
# String-level + macro-expansion checks only; no toolchain required.

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
