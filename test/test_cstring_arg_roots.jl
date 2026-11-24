# =============================================================================
# A string at a `const char*` leaves the wrapper holding no GC root (2026-11-21)
#
# Found by the Hub's lua on the 4.6.0 release sweep: its deep test segfaulted
# in the collector, every run.
#
# Some C functions never return to their caller. `luaL_error`, `png_error`
# and libjpeg's `error_exit` leave by `longjmp`, and a Julia callback calls
# them to raise the library's error. The Julia frames between the jump and
# its landing are abandoned with their GC frames still linked, so the task's
# GC stack points into dead stack memory until the function that made the
# protected call returns, and a collection in that window reads it.
#
# A wrapper on that path must therefore have no GC frame. Before #130 it had
# none: `ccall(…, (Ptr{UInt8},), s)` on a `String` roots nothing. #130 bound a
# `const char*` argument through `JITManager.cstring_arg`, which called Base's
# `Cstring` conversion: inlined, its `throw` builds a message, and that gave
# the whole wrapper a frame.
#
# `cstring_arg(::String)` now inlines to one `memchr` and the argument itself,
# with the refusal out of line. This file reads the compiled wrapper, so it
# needs no toolchain; test_longjmp_wrappers_live.jl makes the jump.
# =============================================================================

using Test
using RepliBuild
using InteractiveUtils: code_llvm

const _CAR_JM = RepliBuild.JITManager
const _CAR_W = RepliBuild.Wrapper

# The conversion both generators and the varargs path emit, taken from the
# generator, inside the two call shapes they emit around it. The library is
# never opened: only compiled code is read.
const _CAR_CONV = _CAR_W._cstring_param_arg("message", "const char*", "Ptr{UInt8}")
const _CAR_MOD = Module(:CstringArgRoots)
Core.eval(_CAR_MOD, :(import RepliBuild))
Core.eval(_CAR_MOD, Meta.parse("""
function plain(message::Any)::Cvoid
    message_c = $(_CAR_CONV)
    return ccall((:lj_fail, "libnowhere"), Cvoid, (Ptr{UInt8},), message_c)
end"""))
Core.eval(_CAR_MOD, Meta.parse("""
function variadic(message::Any, va_1::Integer)::Cint
    message_c = $(_CAR_CONV)
    return @ccall "libnowhere".lj_failf(message_c::Ptr{UInt8}; va_1::Cint)::Cint
end"""))
# A `char*` that is not `const` (2026-11-23): a buffer, where a string is
# refused. The wrapper keeps an integer beside it, as `cs_copy`'s `cap`.
const _CAR_BUF = _CAR_W._cstring_param_arg("dst", "char*", "Ptr{UInt8}")
Core.eval(_CAR_MOD, Meta.parse("""
function writes(dst::Any, cap::Integer)::Cvoid
    dst_c = $(_CAR_BUF)
    cap_c = Csize_t(cap)
    return ccall((:lj_fill, "libnowhere"), Cvoid, (Ptr{UInt8}, Csize_t), dst_c, cap_c)
end"""))
# The same call shape with the `const char*` conversion: what a `Vector`
# costs there already, the baseline the buffer may not exceed.
Core.eval(_CAR_MOD, Meta.parse("""
function writes_before(dst::Any, cap::Integer)::Cvoid
    dst_c = $(_CAR_CONV)
    cap_c = Csize_t(cap)
    return ccall((:lj_fill, "libnowhere"), Cvoid, (Ptr{UInt8}, Csize_t), dst_c, cap_c)
end"""))
# What a wrapper was before the conversion existed.
Core.eval(_CAR_MOD, :(bare(message::Any)::Cvoid =
    ccall((:lj_fail, "libnowhere"), Cvoid, (Ptr{UInt8},), message)))

_car_ir(f, T) = sprint(io -> code_llvm(io, f, T; debuginfo = :none))
# Julia's GC lowering names the frame it allocates.
_car_frame(f, T) = occursin("gcframe", _car_ir(f, T))

@testset "A string argument on a ccall path holds no GC root" begin

    @testset "the generator emits the conversion this file compiles" begin
        @test _CAR_CONV == "RepliBuild.JITManager.cstring_arg(message)"
    end

    @testset "a String is passed as itself" begin
        s = "left by longjmp"
        @test _CAR_JM.cstring_arg(s) === s
        @test _CAR_JM.cstring_arg("") === ""
        # A string of another type is still copied, which terminates it.
        sub = SubString("hello world", 1, 5)
        c = _CAR_JM.cstring_arg(sub)
        @test c isa String && c == "hello"
        # What is not a string passes as it came.
        v = UInt8[0x61, 0x00]
        @test _CAR_JM.cstring_arg(v) === v
        @test _CAR_JM.cstring_arg(C_NULL) === C_NULL
        cu = codeunits("a\0b")
        @test _CAR_JM.cstring_arg(cu) === cu
    end

    @testset "an embedded NUL is refused, and the message says what to pass" begin
        for bad in ("ab\0cd", SubString("xab\0cd", 2), "\0")
            err = try
                _CAR_JM.cstring_arg(bad)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            msg = err isa ArgumentError ? err.msg : ""
            @test occursin("embedded NULs are not allowed in C strings", msg)
            @test occursin("codeunits", msg)
        end
        @test _CAR_JM.cstring_arg("no nul") == "no nul"
    end

    @testset "the wrapper compiled for a String has no GC frame" begin
        # The detector sees a frame where there has to be one: a SubString's
        # copy is rooted across the call. Without this the checks below would
        # pass on a Julia that names the frame differently.
        @test _car_frame(_CAR_MOD.plain, (SubString{String},))

        @test !_car_frame(_CAR_MOD.bare, (String,))
        @test !_car_frame(_CAR_MOD.plain, (String,))
        @test !_car_frame(_CAR_MOD.variadic, (String, Int))
        # Pointers and C strings never had one.
        @test !_car_frame(_CAR_MOD.plain, (Ptr{UInt8},))
        @test !_car_frame(_CAR_MOD.plain, (Cstring,))
    end

    @testset "a char* that is not const gives a pointer, a buffer or an integer nothing new to hold" begin
        @test _CAR_BUF == "RepliBuild.JITManager.char_buffer_arg(dst)"
        @test !_car_frame(_CAR_MOD.writes, (Ptr{UInt8}, Int))
        @test !_car_frame(_CAR_MOD.writes, (Ptr{Cvoid}, Int32))
        # A `Vector`'s memory is rooted across any `ccall` by Base; the buffer
        # conversion adds nothing to that.
        @test _car_frame(_CAR_MOD.writes, (Vector{UInt8}, Int)) == _car_frame(_CAR_MOD.writes_before, (Vector{UInt8}, Int))
        # A string never reaches the call: the refusal is a call, out of line,
        # and nothing after it runs.
        ir = _car_ir(_CAR_MOD.writes, (String, Int))
        @test occursin("char_buffer_arg", ir) || occursin("_refuse_string_at_char_buffer", ir)
        @test occursin("unreachable", ir) && !occursin("lj_fill", ir)
        @test !_car_frame(_CAR_MOD.writes, (String, Int))
        @test !occursin("gc_pool_alloc", ir) && !occursin("gc_small_alloc", ir) && !occursin("jl_alloc", ir)
    end

    @testset "the refusal is a call, not inlined code" begin
        ir = _car_ir(_CAR_MOD.plain, (String,))
        @test occursin("_refuse_embedded_nul", ir)
        @test occursin("memchr", ir)
        # The message is built in the callee: nothing here allocates.
        @test !occursin("gc_pool_alloc", ir) && !occursin("gc_small_alloc", ir) &&
              !occursin("jl_alloc", ir)
    end
end
