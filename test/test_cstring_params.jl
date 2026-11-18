# =============================================================================
# Pointers to `char` on the ccall paths, and `signed char*` returns (2026-11-18)
#
# Two defects of the 2026-11-17 audit, toolchain-free here (the live twin,
# test_cstring_params_live.jl, builds and calls both):
#
#   B16  A `const char*` parameter on a `ccall` path (every function of a C
#        wrapper; a C++ wrapper's `noexcept` and proven-`nounwind` C-linkage
#        functions) was `Ptr{UInt8}` with the argument passed as it came, so
#        `my_strlen(SubString("hello world", 1, 5))` read the parent to its
#        end (11) and `my_strlen("ab\0cd")` was 2 with no error. Tier 2 had
#        Julia's `Cstring` semantics already (`pointer_arg(Cstring, s)`).
#        Every ccall path now binds such an argument through
#        `JITManager.cstring_arg`, whose string arm is the one `pointer_arg`
#        uses, and decides which parameters with `_points_to_plain_char`.
#   B18  `const signed char *f(void)` returned a `String`: the build mapped
#        `signed char*` to `Cstring` like `char*`. Only plain `char` is text;
#        `signed char*` is `Ptr{Int8}`, from the build, and from metadata
#        written before through `_signed_char_returns!`.
# =============================================================================

using Test
using RepliBuild

const _CSP_W = RepliBuild.Wrapper
const _CSP_JM = RepliBuild.JITManager

@testset "Pointers to char on the ccall paths" begin

    @testset "which C spellings are text: _points_to_plain_char" begin
        text = ["char*", "const char*", "char const*", "const char* const", "volatile char*",
                "const volatile char *"]
        data = ["signed char*", "const signed char*", "unsigned char*", "const unsigned char*",
                "int8_t*", "uint8_t*", "char**", "const const char**", "char&", "const char&",
                "char", "", "void*", "wchar_t*", "char16_t*", "cstr_t"]
        for t in text
            @test (t, _CSP_W._points_to_plain_char(t)) == (t, true)
        end
        for t in data
            @test (t, _CSP_W._points_to_plain_char(t)) == (t, false)
        end
    end

    @testset "what a ccall path binds an argument to: _cstring_param_arg" begin
        conv = "RepliBuild.JITManager.cstring_arg(s)"
        @test _CSP_W._cstring_param_arg("s", "const char*", "Ptr{UInt8}") == conv
        @test _CSP_W._cstring_param_arg("s", "char*", "Ptr{UInt8}") == conv
        @test _CSP_W._cstring_param_arg("s", "const char*", "Ptr{Cchar}") == conv
        # A byte pointer that is not text, and a char* the ccall does not
        # declare as a byte pointer, are left alone.
        @test _CSP_W._cstring_param_arg("s", "const unsigned char*", "Ptr{UInt8}") === nothing
        @test _CSP_W._cstring_param_arg("s", "const signed char*", "Ptr{Cchar}") === nothing
        @test _CSP_W._cstring_param_arg("s", "char**", "Ptr{Ptr{UInt8}}") === nothing
        @test _CSP_W._cstring_param_arg("s", "const char*", "Ptr{Cvoid}") === nothing
        @test _CSP_W._cstring_param_arg("s", "", "Ptr{UInt8}") === nothing
    end

    @testset "JITManager.cstring_arg: Julia's Cstring semantics for a string, the rest as it came" begin
        sub = SubString("hello world", 1, 5)
        c = _CSP_JM.cstring_arg(sub)
        # A copy, so the bytes after it are a NUL and not " world".
        @test c isa String && c == "hello"
        @test GC.@preserve c unsafe_load(pointer(c), 6) == 0x00
        @test_throws ArgumentError _CSP_JM.cstring_arg("ab\0cd")
        @test_throws ArgumentError _CSP_JM.cstring_arg(SubString("xab\0cd", 2))
        s = "plain"
        @test _CSP_JM.cstring_arg(s) === s
        buf = UInt8[0x61, 0x00, 0x62]       # bytes, embedded NUL and all
        @test _CSP_JM.cstring_arg(buf) === buf
        sbuf = Int8[97, 0]
        @test _CSP_JM.cstring_arg(sbuf) === sbuf
        @test _CSP_JM.cstring_arg(C_NULL) === C_NULL
        r = Ref{UInt8}(0)
        @test _CSP_JM.cstring_arg(r) === r
        @test _CSP_JM.cstring_arg(:sym) === :sym
        GC.@preserve s begin
            cs = Base.unsafe_convert(Cstring, s)
            @test _CSP_JM.cstring_arg(cs) == pointer(cs)
        end
        # Tier 2 gives a string the same answer through `pointer_arg`.
        @test_throws ArgumentError _CSP_JM.pointer_arg(Cstring, "ab\0cd")
        pa = _CSP_JM.pointer_arg(Cstring, sub)
        @test pa.keep == "hello" && pa.keep isa String
    end

    @testset "the [wrap.varargs] fixed parameters take the same conversion" begin
        params = [
            Dict{String,Any}("name" => "fmt", "julia_type" => "Ptr{UInt8}", "c_type" => "const char*"),
            Dict{String,Any}("name" => "raw", "julia_type" => "Ptr{UInt8}", "c_type" => "const unsigned char*"),
            Dict{String,Any}("name" => "varargs...", "julia_type" => ""),
        ]
        ret = Dict{String,Any}("julia_type" => "Cint", "c_type" => "int")
        for lang in (:c, :cpp)
            code, _ = _CSP_W.generate_vararg_wrappers("vfmt", "vfmt", "vfmt", params, ret, [["Cint"]],
                                                      false, "vfmt(const char*, const unsigned char*, ...)", lang)
            @test occursin("fmt_c = RepliBuild.JITManager.cstring_arg(fmt)", code)
            @test occursin("fmt_c::Ptr{UInt8}", code)
            @test !occursin("cstring_arg(raw)", code)
            parsed = Meta.parseall(code)
            @test !any(ex -> ex isa Expr && ex.head in (:error, :incomplete), parsed.args)
        end
    end

    @testset "signed char* is bytes: the build's map" begin
        d2j = RepliBuild.Compiler.dwarf_type_to_julia
        @test d2j("const signed char*") == "Ptr{Int8}"
        @test d2j("signed char*") == "Ptr{Int8}"
        @test d2j("signed char**") == "Ptr{Ptr{Int8}}"
        # Unchanged: plain char is text, unsigned char is bytes.
        @test d2j("const char*") == "Cstring"
        @test d2j("char*") == "Cstring"
        @test d2j("char**") == "Ptr{Cstring}"
        @test d2j("const unsigned char*") == "Ptr{Cuchar}"
    end

    @testset "signed char* is bytes: metadata written before (_signed_char_returns!)" begin
        fn(name, c_type, jt) = Dict{String,Any}("name" => name, "parameters" => Any[],
            "return_type" => Dict{String,Any}("c_type" => c_type, "julia_type" => jt, "size" => 8))
        meta = Dict{String,Any}("functions" => Any[
            fn("sbytes", "const signed char*", "Cstring"),
            fn("sbytes2", "signed char *", "Cstring"),
            fn("text", "const char*", "Cstring"),
            fn("typedef_text", "cstr_t", "Cstring"),     # a typedef: keeps the policy
            fn("ubytes", "const unsigned char*", "Ptr{Cuchar}"),
            fn("already", "const signed char*", "Ptr{Int8}"),
        ])
        @test _CSP_W._signed_char_returns!(meta) == 2
        jt = Dict(f["name"] => f["return_type"]["julia_type"] for f in meta["functions"])
        @test jt == Dict("sbytes" => "Ptr{Int8}", "sbytes2" => "Ptr{Int8}", "text" => "Cstring",
                         "typedef_text" => "Cstring", "ubytes" => "Ptr{Cuchar}", "already" => "Ptr{Int8}")
        @test _CSP_W._signed_char_returns!(meta) == 0
    end
end
