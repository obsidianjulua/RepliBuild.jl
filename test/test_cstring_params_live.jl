# =============================================================================
# Pointers to `char` on the ccall paths, live (2026-11-18)
#
# test_cstring_params.jl checks the derivations. This builds two fixtures,
# wraps them and calls them:
#
#   test/cstring_param_c_test (C: every function a ccall)
#     - `cs_len(SubString("hello world", 1, 5))` was 11 (the parent read to its
#       end) and `cs_len("ab\0cd")` was 2 with no error: the `const char*` was
#       a `Ptr{UInt8}` the string converted to as it came. A string at a
#       pointer to plain `char` now has Julia's `Cstring` semantics, and a
#       byte buffer, a pointer, a `Ref` pass as before;
#     - the fixed `const char*` of a `[wrap.varargs]` function, the same;
#     - `const signed char *cs_sbytes(void)` returned the `String` "\xff\xfe".
#       It is `Ptr{Int8}`, with no `_ptr` sibling, and stays so when the
#       metadata on disk predates the build's fix (wrapped again from
#       metadata that says `Cstring`).
#
#     - a `char*` that is not `const` (`cs_upcase`, `cs_copy`'s `dst`) is a
#       buffer the library writes: a Julia string there is an
#       `ArgumentError` (2026-11-23), a byte buffer, a `Ptr`, a `Ref` pass.
#
#   test/cstring_param_cpp_test (C++)
#     - a `noexcept` function and an `extern "C"` function proven `nounwind`
#       are ccalls and get the same semantics as the Tier 2 function beside
#       them; `signed char*` returns are `Ptr{Int8}` on both paths.
#
# Calls stay in this process: the old wrappers answered wrong values, they did
# not crash. Needs the C bucket; the C++ half needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const CSTR_C_DIR = joinpath(@__DIR__, "cstring_param_c_test")
const CSTR_CPP_DIR = joinpath(@__DIR__, "cstring_param_cpp_test")

function cstr_body(wsrc::AbstractString, name::AbstractString)
    m = match(Regex("^function $(name)\\(.*?^end\$", "ms"), wsrc)
    return m === nothing ? "" : String(m.match)
end

function cstr_build(dir, name, language, flags, src; extra = "")
    for d in ("build", "julia", ".replibuild_cache", ".debug")
        p = joinpath(dir, d)
        ispath(p) && RepliBuild._rm_tree(p)
    end
    toml_path = joinpath(dir, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "$name"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(dir))"

    [compile]
    flags = [$(join(repr.(flags), ", "))]
    source_files = ["$(escape_string(src))"]
    include_dirs = ["$(escape_string(dir))"]

    [link]
    enable_lto = false
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    language = "$language"
    $extra
    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    redirect_stdout(devnull) do
        RepliBuild.build(toml_path)
    end
    RepliBuild.wrap(toml_path)
    return toml_path
end

# A string at a pointer to plain char, through one wrapper function `f`
# taking it alone and answering its length.
function cstr_semantics(f)
    @test f("hello") == 5
    @test f(SubString("hello world", 1, 5)) == 5
    @test f(SubString("xxhello", 3)) == 5
    @test_throws ArgumentError f("ab\0cd")
    @test f(UInt8[0x61, 0x62, 0x00]) == 2
    @test f(Int8[97, 98, 99, 0]) == 3
    s = "abcd"
    GC.@preserve s begin
        @test f(pointer(s)) == 4
        @test f(Base.unsafe_convert(Cstring, s)) == 4
    end
    @test f(Ref((0x61, 0x00))) == 1
end

# A `char*` that is not `const`, through one wrapper function `f` that
# upper-cases what it is given in place (2026-11-23). A Julia string is
# refused, and stays as it was; a byte buffer is written.
function char_buffer_semantics(f)
    for s in ("abc", SubString("xabc", 2), :abc, codeunits("abc"), codeunits(SubString("xabc", 2)))
        err = try
            f(s); nothing
        catch e
            e
        end
        @test (typeof(s), err isa ArgumentError) == (typeof(s), true)
        msg = err isa ArgumentError ? err.msg : ""
        @test occursin("`char *` it may write into", msg) && occursin("Vector{UInt8}(\"text\\0\")", msg)
    end
    lit() = (s = "abc"; try f(s) catch end; s)
    @test lit() == "abc" && lit() == "abc"
    buf = Vector{UInt8}("abc\0")
    f(buf)
    @test GC.@preserve(buf, unsafe_string(pointer(buf))) == "ABC"
    sbuf = Int8[104, 105, 0]
    f(sbuf)
    @test sbuf == Int8[72, 73, 0]
    GC.@preserve buf begin
        buf2 = Vector{UInt8}("xyz\0")
        GC.@preserve buf2 f(pointer(buf2))
        @test String(buf2[1:3]) == "XYZ"
    end
    r = Ref((UInt8('q'), 0x00))
    f(r)
    @test r[][1] == UInt8('Q')
end

# One top-level testset, so a red C half does not hide the C++ half.
@testset "Pointers to char, live" begin
@testset "Pointers to char, live: C" begin
    toml_path = cstr_build(CSTR_C_DIR, "cstring_param_c", "c", ["-O0", "-fPIC"],
                           joinpath(CSTR_C_DIR, "cstrings.c");
                           extra = "\n[wrap.varargs]\ncs_fmt_len = [[\"Cint\", \"Cint\"]]\n")
    wrapper = joinpath(CSTR_C_DIR, "julia", "CstringParamC.jl")
    @test isfile(wrapper)
    wsrc = isfile(wrapper) ? read(wrapper, String) : ""

    @testset "what was emitted" begin
        @test occursin("RepliBuild.JITManager.cstring_arg(s)", cstr_body(wsrc, "cs_len"))
        @test occursin("RepliBuild.JITManager.cstring_arg(src)", cstr_body(wsrc, "cs_copy"))
        @test occursin("RepliBuild.JITManager.char_buffer_arg(dst)", cstr_body(wsrc, "cs_copy"))
        @test occursin("RepliBuild.JITManager.char_buffer_arg(s)", cstr_body(wsrc, "cs_upcase"))
        @test !occursin("cstring_arg", cstr_body(wsrc, "cs_sum_ubytes"))
        @test occursin("RepliBuild.JITManager.cstring_arg(fmt)", cstr_body(wsrc, "cs_fmt_len"))
        @test occursin("RepliBuild.JITManager.cstring_arg(fmt)", cstr_body(wsrc, "cs_fmt_len_Cint_Cint"))
        @test occursin(r"^function cs_sbytes\(\)::Ptr\{Int8\}"m, wsrc)
        @test !occursin("cs_sbytes_ptr", wsrc)
        @test occursin(r"^function cs_greet\(\)::Union\{String,Nothing\}"m, wsrc)
        @test occursin(r"^function cs_greet_ptr\("m, wsrc)
        meta = RepliBuild._read_json(joinpath(CSTR_C_DIR, "julia", "compilation_metadata.json"))
        rets = Dict(String(f["name"]) => f["return_type"]["julia_type"] for f in meta["functions"])
        @test rets["cs_sbytes"] == "Ptr{Int8}"
        @test rets["cs_greet"] == "Cstring"
    end

    @testset "calls" begin
        M = Base.include(Module(), wrapper)
        cstr_semantics(M.cs_len)
        # A buffer the library writes into is passed as it came.
        b = zeros(UInt8, 8)
        @test M.cs_copy(b, 8, SubString("heyyou", 1, 3)) == 3 && String(b[1:3]) == "hey" && b[4] == 0
        # A byte pointer that is not text: a string's bytes, NUL and all.
        @test M.cs_sum_ubytes("a\0b", 3) == 97 + 98
        @test M.cs_sum_ubytes(UInt8[1, 2, 3], 3) == 6
        p = M.cs_sbytes()
        @test p isa Ptr{Int8} && unsafe_load(p, 1) == -1 && unsafe_load(p, 2) == -2
        @test M.cs_ubytes() isa Ptr{UInt8}
        @test M.cs_greet() == "hello"
        @test M.cs_fmt_len(SubString("abcdef", 1, 2), 0) == 2
        @test M.cs_fmt_len_Cint_Cint(SubString("abcdef", 1, 3), 2, 10, 20) == 33
        @test_throws ArgumentError M.cs_fmt_len("a\0b", 0)
        char_buffer_semantics(M.cs_upcase)
        # A string where the library writes (`cs_copy`'s `dst`) is refused too.
        @test_throws ArgumentError M.cs_copy("12345678", 8, "hey")
    end

    @testset "metadata from before the build's fix" begin
        # The `signed char*` return as the old build recorded it: `Cstring`.
        mpath = joinpath(CSTR_C_DIR, "julia", "compilation_metadata.json")
        meta = RepliBuild._read_json(mpath)
        n = 0
        for f in meta["functions"]
            f["name"] == "cs_sbytes" || continue
            f["return_type"]["julia_type"] = "Cstring"
            n += 1
        end
        @test n == 1
        write(mpath, RepliBuild.JSON.json(meta))
        RepliBuild.wrap(toml_path)
        wsrc2 = read(wrapper, String)
        @test occursin(r"^function cs_sbytes\(\)::Ptr\{Int8\}"m, wsrc2)
        @test !occursin("cs_sbytes_ptr", wsrc2)
        @test occursin(r"^function cs_greet\(\)::Union\{String,Nothing\}"m, wsrc2)
    end
end

if requires("pointers to char through a C++ wrapper, live", :libJLCS, :clangxx)
@testset "Pointers to char, live: C++ (ccall paths beside Tier 2)" begin
    cstr_build(CSTR_CPP_DIR, "cstring_param_cpp", "cpp", ["-std=c++17", "-O0", "-fPIC"],
               joinpath(CSTR_CPP_DIR, "cstrings.cpp"))
    wrapper = joinpath(CSTR_CPP_DIR, "julia", "CstringParamCpp.jl")
    @test isfile(wrapper)
    wsrc = isfile(wrapper) ? read(wrapper, String) : ""

    @testset "what was emitted" begin
        meta = RepliBuild._read_json(joinpath(CSTR_CPP_DIR, "julia", "compilation_metadata.json"))
        by_name = Dict(String(f["name"]) => f for f in meta["functions"])
        # `c_len` reaches ccall only when proven `nounwind`: say so, or the
        # check below is about the thunk.
        @test get(by_name["c_len"], "nounwind", nothing) == true
        @test by_name["cpp_sbytes"]["return_type"]["julia_type"] == "Ptr{Int8}"
        for f in ("cpp_len", "c_len")
            b = cstr_body(wsrc, f)
            @test (f, occursin("ccall((:", b)) == (f, true)
            @test (f, occursin("RepliBuild.JITManager.cstring_arg(s)", b)) == (f, true)
        end
        @test occursin("RepliBuild.JITManager.pointer_arg(Cstring, s)", cstr_body(wsrc, "thunk_len"))
        @test occursin("ccall((:", cstr_body(wsrc, "cpp_upcase"))
        @test occursin("RepliBuild.JITManager.char_buffer_arg(s)", cstr_body(wsrc, "cpp_upcase"))
        @test occursin("RepliBuild.JITManager.pointer_arg(RepliBuild.JITManager.CharBuffer, s)",
                       cstr_body(wsrc, "thunk_upcase"))
        @test occursin(r"^function cpp_sbytes\(\)::Ptr\{Int8\}"m, wsrc)
        @test occursin("ccall((:", cstr_body(wsrc, "cpp_sbytes"))
        @test occursin("_Z12thunk_sbytesi_thunk\", Ptr{Int8}, ", cstr_body(wsrc, "thunk_sbytes"))
        @test !occursin("cpp_sbytes_ptr", wsrc) && !occursin("thunk_sbytes_ptr", wsrc)
        @test occursin(r"^function cpp_greet\(\)::Union\{String,Nothing\}"m, wsrc)
    end

    @testset "calls" begin
        M = Base.include(Module(), wrapper)
        cstr_semantics(M.cpp_len)
        cstr_semantics(M.c_len)
        cstr_semantics(M.thunk_len)
        char_buffer_semantics(M.cpp_upcase)
        char_buffer_semantics(M.thunk_upcase)
        for p in (M.cpp_sbytes(), M.thunk_sbytes(0))
            @test p isa Ptr{Int8} && unsafe_load(p, 1) == -1
        end
        @test M.cpp_greet() == "hello"
    end
end
end # requires
end
