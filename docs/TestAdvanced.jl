# Auto-generated Julia wrapper for TestAdvanced
# Generated: 2025-11-25 04:03:19
# Generator: RepliBuild Wrapper (Tier 3: Introspective)
# Library: libtest.so
# Metadata: compilation_metadata.json
#
# Type Safety:  Perfect - Types extracted from compilation
# Language: Language-agnostic (via LLVM IR)
# Manual edits: None required

module TestAdvanced

const LIBRARY_PATH = "/home/grim/Desktop/Projects/RepliBuild.jl/test/build_simple/libtest.so"

# Verify library exists
if !isfile(LIBRARY_PATH)
    error("Library not found: $LIBRARY_PATH")
end

# =============================================================================
# Compilation Metadata
# =============================================================================

const METADATA = Dict(
    "llvm_version" => "unknown",
    "clang_version" => "unknown",
    "optimization" => "unknown",
    "target_triple" => "unknown",
    "function_count" => 0,
    "generated_at" => "unknown"
)

# =============================================================================
# Enum Definitions (from DWARF debug info)
# =============================================================================

# C++ enum: Color (underlying type: unsigned int)
@enum Color::Cuint begin
    Red = 0
    Green = 1
    Blue = 2
end

# C++ enum: Status (underlying type: unsigned int)
@enum Status::Cuint begin
    Idle = 0
    Running = 100
    Stopped = 200
    Error = 999
end


# =============================================================================
# Struct Definitions (from DWARF debug info)
# =============================================================================

# C++ struct: ComplexType (5 members)
mutable struct ComplexType
    color::Any
    status::Any
    coords::NTuple{3, Cdouble}
    handler::Any
    matrix::NTuple{6, Cint}
end

# C++ struct: Grid (2 members)
mutable struct Grid
    cells::NTuple{16, Cint}
    values::NTuple{3, Cdouble}
end

# C++ struct: Matrix3x3 (1 members)
mutable struct Matrix3x3
    data::NTuple{9, Cdouble}
end


export acosl, powl, nexttowardl, llround, asin, nearbyint, erfcf, round, remainderf, floor, lgammaf, roundl, asinh, atan2l, tanhf, fabs, ilogbf, exp, expm1f, cbrtl, remquo, cbrt, atanhl, acosf, fmal, llrintf, coshf, nearbyintf, tgammal, cosf, hypot, erff, _Z10matrix_sum9Matrix3x3, cosh, scalblnf, lround, fmaf, copysign, fminl, pow, hypotl, sinf, erfl, erfc, scalbln, acoshl, nexttoward, fabsf, atan2, fmin, lrintl, log1p, atanh, floorf, copysignf, expf, cbrtf, ilogbl, frexpl, llrintl, asinf, nexttowardf, tanl, log10f, ldexpl, truncl, _Z12add_callbackdd, sqrt, lrintf, log2l, fdim, fmod, log2f, lroundf, ceilf, lrint, log, trunc, acos, scalbnl, asinhl, powf, fmodf, remainderl, abs, log1pl, fdiml, modf, cos, erfcl, remquof, fma, coshl, logl, tanhl, asinhf, _Z12check_status6Status, remainder, lgammal, log2, lroundl, tanf, asinl, nextafterl, fmaxl, acosh, llrint, scalbn, exp2l, frexpf, copysignl, rintf, roundf, log1pf, lgamma, _Z9run_testsv, _Z17get_primary_colorv, fminf, sinh, ceil, llroundf, expm1, atanhf, llroundl, nearbyintl, tgamma, cosl, expm1l, ceill, exp2f, sinhf, expl, tan, rintl, floorl, ldexpf, frexp, nanf, rint, atanf, _Z14apply_callbackPFiddEdd, logf, ilogb, hypotf, nan, fmax, acoshf, atan2f, nanl, ldexp, atanl, truncf, exp2, atan, _Z12color_to_int5Color, fmaxf, nextafter, nextafterf, remquol, sinhl, _Z22create_identity_matrixv, tanh, log10l, _Z8grid_get4Gridii, scalblnl, fmodl, log10, logbl, erf, fabsl, modff, sqrtf, scalbnf, logb, tgammaf, sqrtl, sinl, logbf, modfl, _Z14create_complex5Color6StatusPFiddE, fdimf, sin

"""
    acosl() -> Float64

Wrapper for C++ function: `acosl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `acosl`
- Type safety:  From compilation
"""

function acosl()::Float64
    ccall((:acosl, LIBRARY_PATH), Float64, (), )
end

"""
    powl() -> Float64

Wrapper for C++ function: `powl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `powl`
- Type safety:  From compilation
"""

function powl()::Float64
    ccall((:powl, LIBRARY_PATH), Float64, (), )
end

"""
    nexttowardl() -> Float64

Wrapper for C++ function: `nexttowardl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `nexttowardl`
- Type safety:  From compilation
"""

function nexttowardl()::Float64
    ccall((:nexttowardl, LIBRARY_PATH), Float64, (), )
end

"""
    llround() -> Clonglong

Wrapper for C++ function: `llround`

# Arguments


# Returns
- `Clonglong`

# Metadata
- Mangled symbol: `llround`
- Type safety:  From compilation
"""

function llround()::Clonglong
    ccall((:llround, LIBRARY_PATH), Clonglong, (), )
end

"""
    asin() -> Cdouble

Wrapper for C++ function: `asin`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `asin`
- Type safety:  From compilation
"""

function asin()::Cdouble
    ccall((:asin, LIBRARY_PATH), Cdouble, (), )
end

"""
    nearbyint() -> Cdouble

Wrapper for C++ function: `nearbyint`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `nearbyint`
- Type safety:  From compilation
"""

function nearbyint()::Cdouble
    ccall((:nearbyint, LIBRARY_PATH), Cdouble, (), )
end

"""
    erfcf() -> Cfloat

Wrapper for C++ function: `erfcf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `erfcf`
- Type safety:  From compilation
"""

function erfcf()::Cfloat
    ccall((:erfcf, LIBRARY_PATH), Cfloat, (), )
end

"""
    round() -> Cdouble

Wrapper for C++ function: `round`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `round`
- Type safety:  From compilation
"""

function round()::Cdouble
    ccall((:round, LIBRARY_PATH), Cdouble, (), )
end

"""
    remainderf() -> Cfloat

Wrapper for C++ function: `remainderf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `remainderf`
- Type safety:  From compilation
"""

function remainderf()::Cfloat
    ccall((:remainderf, LIBRARY_PATH), Cfloat, (), )
end

"""
    floor() -> Cdouble

Wrapper for C++ function: `floor`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `floor`
- Type safety:  From compilation
"""

function floor()::Cdouble
    ccall((:floor, LIBRARY_PATH), Cdouble, (), )
end

"""
    lgammaf() -> Cfloat

Wrapper for C++ function: `lgammaf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `lgammaf`
- Type safety:  From compilation
"""

function lgammaf()::Cfloat
    ccall((:lgammaf, LIBRARY_PATH), Cfloat, (), )
end

"""
    roundl() -> Float64

Wrapper for C++ function: `roundl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `roundl`
- Type safety:  From compilation
"""

function roundl()::Float64
    ccall((:roundl, LIBRARY_PATH), Float64, (), )
end

"""
    asinh() -> Cdouble

Wrapper for C++ function: `asinh`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `asinh`
- Type safety:  From compilation
"""

function asinh()::Cdouble
    ccall((:asinh, LIBRARY_PATH), Cdouble, (), )
end

"""
    atan2l() -> Float64

Wrapper for C++ function: `atan2l`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `atan2l`
- Type safety:  From compilation
"""

function atan2l()::Float64
    ccall((:atan2l, LIBRARY_PATH), Float64, (), )
end

"""
    tanhf() -> Cfloat

Wrapper for C++ function: `tanhf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `tanhf`
- Type safety:  From compilation
"""

function tanhf()::Cfloat
    ccall((:tanhf, LIBRARY_PATH), Cfloat, (), )
end

"""
    fabs() -> Cdouble

Wrapper for C++ function: `fabs`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `fabs`
- Type safety:  From compilation
"""

function fabs()::Cdouble
    ccall((:fabs, LIBRARY_PATH), Cdouble, (), )
end

"""
    ilogbf() -> Cint

Wrapper for C++ function: `ilogbf`

# Arguments


# Returns
- `Cint`

# Metadata
- Mangled symbol: `ilogbf`
- Type safety:  From compilation
"""

function ilogbf()::Cint
    ccall((:ilogbf, LIBRARY_PATH), Cint, (), )
end

"""
    exp() -> Cdouble

Wrapper for C++ function: `exp`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `exp`
- Type safety:  From compilation
"""

function exp()::Cdouble
    ccall((:exp, LIBRARY_PATH), Cdouble, (), )
end

"""
    expm1f() -> Cfloat

Wrapper for C++ function: `expm1f`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `expm1f`
- Type safety:  From compilation
"""

function expm1f()::Cfloat
    ccall((:expm1f, LIBRARY_PATH), Cfloat, (), )
end

"""
    cbrtl() -> Float64

Wrapper for C++ function: `cbrtl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `cbrtl`
- Type safety:  From compilation
"""

function cbrtl()::Float64
    ccall((:cbrtl, LIBRARY_PATH), Float64, (), )
end

"""
    remquo() -> Cdouble

Wrapper for C++ function: `remquo`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `remquo`
- Type safety:  From compilation
"""

function remquo()::Cdouble
    ccall((:remquo, LIBRARY_PATH), Cdouble, (), )
end

"""
    cbrt() -> Cdouble

Wrapper for C++ function: `cbrt`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `cbrt`
- Type safety:  From compilation
"""

function cbrt()::Cdouble
    ccall((:cbrt, LIBRARY_PATH), Cdouble, (), )
end

"""
    atanhl() -> Float64

Wrapper for C++ function: `atanhl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `atanhl`
- Type safety:  From compilation
"""

function atanhl()::Float64
    ccall((:atanhl, LIBRARY_PATH), Float64, (), )
end

"""
    acosf() -> Cfloat

Wrapper for C++ function: `acosf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `acosf`
- Type safety:  From compilation
"""

function acosf()::Cfloat
    ccall((:acosf, LIBRARY_PATH), Cfloat, (), )
end

"""
    fmal() -> Float64

Wrapper for C++ function: `fmal`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `fmal`
- Type safety:  From compilation
"""

function fmal()::Float64
    ccall((:fmal, LIBRARY_PATH), Float64, (), )
end

"""
    llrintf() -> Clonglong

Wrapper for C++ function: `llrintf`

# Arguments


# Returns
- `Clonglong`

# Metadata
- Mangled symbol: `llrintf`
- Type safety:  From compilation
"""

function llrintf()::Clonglong
    ccall((:llrintf, LIBRARY_PATH), Clonglong, (), )
end

"""
    coshf() -> Cfloat

Wrapper for C++ function: `coshf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `coshf`
- Type safety:  From compilation
"""

function coshf()::Cfloat
    ccall((:coshf, LIBRARY_PATH), Cfloat, (), )
end

"""
    nearbyintf() -> Cfloat

Wrapper for C++ function: `nearbyintf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `nearbyintf`
- Type safety:  From compilation
"""

function nearbyintf()::Cfloat
    ccall((:nearbyintf, LIBRARY_PATH), Cfloat, (), )
end

"""
    tgammal() -> Float64

Wrapper for C++ function: `tgammal`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `tgammal`
- Type safety:  From compilation
"""

function tgammal()::Float64
    ccall((:tgammal, LIBRARY_PATH), Float64, (), )
end

"""
    cosf() -> Cfloat

Wrapper for C++ function: `cosf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `cosf`
- Type safety:  From compilation
"""

function cosf()::Cfloat
    ccall((:cosf, LIBRARY_PATH), Cfloat, (), )
end

"""
    hypot() -> Cdouble

Wrapper for C++ function: `hypot`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `hypot`
- Type safety:  From compilation
"""

function hypot()::Cdouble
    ccall((:hypot, LIBRARY_PATH), Cdouble, (), )
end

"""
    erff() -> Cfloat

Wrapper for C++ function: `erff`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `erff`
- Type safety:  From compilation
"""

function erff()::Cfloat
    ccall((:erff, LIBRARY_PATH), Cfloat, (), )
end

"""
    _Z10matrix_sum9Matrix3x3() -> Cdouble

Wrapper for C++ function: `_Z10matrix_sum9Matrix3x3`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `_Z10matrix_sum9Matrix3x3`
- Type safety:  From compilation
"""

function _Z10matrix_sum9Matrix3x3()::Cdouble
    ccall((:_Z10matrix_sum9Matrix3x3, LIBRARY_PATH), Cdouble, (), )
end

"""
    cosh() -> Cdouble

Wrapper for C++ function: `cosh`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `cosh`
- Type safety:  From compilation
"""

function cosh()::Cdouble
    ccall((:cosh, LIBRARY_PATH), Cdouble, (), )
end

"""
    scalblnf() -> Cfloat

Wrapper for C++ function: `scalblnf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `scalblnf`
- Type safety:  From compilation
"""

function scalblnf()::Cfloat
    ccall((:scalblnf, LIBRARY_PATH), Cfloat, (), )
end

"""
    lround() -> Clong

Wrapper for C++ function: `lround`

# Arguments


# Returns
- `Clong`

# Metadata
- Mangled symbol: `lround`
- Type safety:  From compilation
"""

function lround()::Clong
    ccall((:lround, LIBRARY_PATH), Clong, (), )
end

"""
    fmaf() -> Cfloat

Wrapper for C++ function: `fmaf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `fmaf`
- Type safety:  From compilation
"""

function fmaf()::Cfloat
    ccall((:fmaf, LIBRARY_PATH), Cfloat, (), )
end

"""
    copysign() -> Cdouble

Wrapper for C++ function: `copysign`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `copysign`
- Type safety:  From compilation
"""

function copysign()::Cdouble
    ccall((:copysign, LIBRARY_PATH), Cdouble, (), )
end

"""
    fminl() -> Float64

Wrapper for C++ function: `fminl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `fminl`
- Type safety:  From compilation
"""

function fminl()::Float64
    ccall((:fminl, LIBRARY_PATH), Float64, (), )
end

"""
    pow() -> Cdouble

Wrapper for C++ function: `pow`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `pow`
- Type safety:  From compilation
"""

function pow()::Cdouble
    ccall((:pow, LIBRARY_PATH), Cdouble, (), )
end

"""
    hypotl() -> Float64

Wrapper for C++ function: `hypotl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `hypotl`
- Type safety:  From compilation
"""

function hypotl()::Float64
    ccall((:hypotl, LIBRARY_PATH), Float64, (), )
end

"""
    sinf() -> Cfloat

Wrapper for C++ function: `sinf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `sinf`
- Type safety:  From compilation
"""

function sinf()::Cfloat
    ccall((:sinf, LIBRARY_PATH), Cfloat, (), )
end

"""
    erfl() -> Float64

Wrapper for C++ function: `erfl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `erfl`
- Type safety:  From compilation
"""

function erfl()::Float64
    ccall((:erfl, LIBRARY_PATH), Float64, (), )
end

"""
    erfc() -> Cdouble

Wrapper for C++ function: `erfc`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `erfc`
- Type safety:  From compilation
"""

function erfc()::Cdouble
    ccall((:erfc, LIBRARY_PATH), Cdouble, (), )
end

"""
    scalbln() -> Cdouble

Wrapper for C++ function: `scalbln`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `scalbln`
- Type safety:  From compilation
"""

function scalbln()::Cdouble
    ccall((:scalbln, LIBRARY_PATH), Cdouble, (), )
end

"""
    acoshl() -> Float64

Wrapper for C++ function: `acoshl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `acoshl`
- Type safety:  From compilation
"""

function acoshl()::Float64
    ccall((:acoshl, LIBRARY_PATH), Float64, (), )
end

"""
    nexttoward() -> Cdouble

Wrapper for C++ function: `nexttoward`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `nexttoward`
- Type safety:  From compilation
"""

function nexttoward()::Cdouble
    ccall((:nexttoward, LIBRARY_PATH), Cdouble, (), )
end

"""
    fabsf() -> Cfloat

Wrapper for C++ function: `fabsf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `fabsf`
- Type safety:  From compilation
"""

function fabsf()::Cfloat
    ccall((:fabsf, LIBRARY_PATH), Cfloat, (), )
end

"""
    atan2() -> Cdouble

Wrapper for C++ function: `atan2`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `atan2`
- Type safety:  From compilation
"""

function atan2()::Cdouble
    ccall((:atan2, LIBRARY_PATH), Cdouble, (), )
end

"""
    fmin() -> Cdouble

Wrapper for C++ function: `fmin`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `fmin`
- Type safety:  From compilation
"""

function fmin()::Cdouble
    ccall((:fmin, LIBRARY_PATH), Cdouble, (), )
end

"""
    lrintl() -> Clong

Wrapper for C++ function: `lrintl`

# Arguments


# Returns
- `Clong`

# Metadata
- Mangled symbol: `lrintl`
- Type safety:  From compilation
"""

function lrintl()::Clong
    ccall((:lrintl, LIBRARY_PATH), Clong, (), )
end

"""
    log1p() -> Cdouble

Wrapper for C++ function: `log1p`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `log1p`
- Type safety:  From compilation
"""

function log1p()::Cdouble
    ccall((:log1p, LIBRARY_PATH), Cdouble, (), )
end

"""
    atanh() -> Cdouble

Wrapper for C++ function: `atanh`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `atanh`
- Type safety:  From compilation
"""

function atanh()::Cdouble
    ccall((:atanh, LIBRARY_PATH), Cdouble, (), )
end

"""
    floorf() -> Cfloat

Wrapper for C++ function: `floorf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `floorf`
- Type safety:  From compilation
"""

function floorf()::Cfloat
    ccall((:floorf, LIBRARY_PATH), Cfloat, (), )
end

"""
    copysignf() -> Cfloat

Wrapper for C++ function: `copysignf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `copysignf`
- Type safety:  From compilation
"""

function copysignf()::Cfloat
    ccall((:copysignf, LIBRARY_PATH), Cfloat, (), )
end

"""
    expf() -> Cfloat

Wrapper for C++ function: `expf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `expf`
- Type safety:  From compilation
"""

function expf()::Cfloat
    ccall((:expf, LIBRARY_PATH), Cfloat, (), )
end

"""
    cbrtf() -> Cfloat

Wrapper for C++ function: `cbrtf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `cbrtf`
- Type safety:  From compilation
"""

function cbrtf()::Cfloat
    ccall((:cbrtf, LIBRARY_PATH), Cfloat, (), )
end

"""
    ilogbl() -> Cint

Wrapper for C++ function: `ilogbl`

# Arguments


# Returns
- `Cint`

# Metadata
- Mangled symbol: `ilogbl`
- Type safety:  From compilation
"""

function ilogbl()::Cint
    ccall((:ilogbl, LIBRARY_PATH), Cint, (), )
end

"""
    frexpl() -> Float64

Wrapper for C++ function: `frexpl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `frexpl`
- Type safety:  From compilation
"""

function frexpl()::Float64
    ccall((:frexpl, LIBRARY_PATH), Float64, (), )
end

"""
    llrintl() -> Clonglong

Wrapper for C++ function: `llrintl`

# Arguments


# Returns
- `Clonglong`

# Metadata
- Mangled symbol: `llrintl`
- Type safety:  From compilation
"""

function llrintl()::Clonglong
    ccall((:llrintl, LIBRARY_PATH), Clonglong, (), )
end

"""
    asinf() -> Cfloat

Wrapper for C++ function: `asinf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `asinf`
- Type safety:  From compilation
"""

function asinf()::Cfloat
    ccall((:asinf, LIBRARY_PATH), Cfloat, (), )
end

"""
    nexttowardf() -> Cfloat

Wrapper for C++ function: `nexttowardf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `nexttowardf`
- Type safety:  From compilation
"""

function nexttowardf()::Cfloat
    ccall((:nexttowardf, LIBRARY_PATH), Cfloat, (), )
end

"""
    tanl() -> Float64

Wrapper for C++ function: `tanl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `tanl`
- Type safety:  From compilation
"""

function tanl()::Float64
    ccall((:tanl, LIBRARY_PATH), Float64, (), )
end

"""
    log10f() -> Cfloat

Wrapper for C++ function: `log10f`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `log10f`
- Type safety:  From compilation
"""

function log10f()::Cfloat
    ccall((:log10f, LIBRARY_PATH), Cfloat, (), )
end

"""
    ldexpl() -> Float64

Wrapper for C++ function: `ldexpl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `ldexpl`
- Type safety:  From compilation
"""

function ldexpl()::Float64
    ccall((:ldexpl, LIBRARY_PATH), Float64, (), )
end

"""
    truncl() -> Float64

Wrapper for C++ function: `truncl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `truncl`
- Type safety:  From compilation
"""

function truncl()::Float64
    ccall((:truncl, LIBRARY_PATH), Float64, (), )
end

"""
    _Z12add_callbackdd() -> Cint

Wrapper for C++ function: `_Z12add_callbackdd`

# Arguments


# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z12add_callbackdd`
- Type safety:  From compilation
"""

function _Z12add_callbackdd()::Cint
    ccall((:_Z12add_callbackdd, LIBRARY_PATH), Cint, (), )
end

"""
    sqrt() -> Cdouble

Wrapper for C++ function: `sqrt`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `sqrt`
- Type safety:  From compilation
"""

function sqrt()::Cdouble
    ccall((:sqrt, LIBRARY_PATH), Cdouble, (), )
end

"""
    lrintf() -> Clong

Wrapper for C++ function: `lrintf`

# Arguments


# Returns
- `Clong`

# Metadata
- Mangled symbol: `lrintf`
- Type safety:  From compilation
"""

function lrintf()::Clong
    ccall((:lrintf, LIBRARY_PATH), Clong, (), )
end

"""
    log2l() -> Float64

Wrapper for C++ function: `log2l`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `log2l`
- Type safety:  From compilation
"""

function log2l()::Float64
    ccall((:log2l, LIBRARY_PATH), Float64, (), )
end

"""
    fdim() -> Cdouble

Wrapper for C++ function: `fdim`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `fdim`
- Type safety:  From compilation
"""

function fdim()::Cdouble
    ccall((:fdim, LIBRARY_PATH), Cdouble, (), )
end

"""
    fmod() -> Cdouble

Wrapper for C++ function: `fmod`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `fmod`
- Type safety:  From compilation
"""

function fmod()::Cdouble
    ccall((:fmod, LIBRARY_PATH), Cdouble, (), )
end

"""
    log2f() -> Cfloat

Wrapper for C++ function: `log2f`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `log2f`
- Type safety:  From compilation
"""

function log2f()::Cfloat
    ccall((:log2f, LIBRARY_PATH), Cfloat, (), )
end

"""
    lroundf() -> Clong

Wrapper for C++ function: `lroundf`

# Arguments


# Returns
- `Clong`

# Metadata
- Mangled symbol: `lroundf`
- Type safety:  From compilation
"""

function lroundf()::Clong
    ccall((:lroundf, LIBRARY_PATH), Clong, (), )
end

"""
    ceilf() -> Cfloat

Wrapper for C++ function: `ceilf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `ceilf`
- Type safety:  From compilation
"""

function ceilf()::Cfloat
    ccall((:ceilf, LIBRARY_PATH), Cfloat, (), )
end

"""
    lrint() -> Clong

Wrapper for C++ function: `lrint`

# Arguments


# Returns
- `Clong`

# Metadata
- Mangled symbol: `lrint`
- Type safety:  From compilation
"""

function lrint()::Clong
    ccall((:lrint, LIBRARY_PATH), Clong, (), )
end

"""
    log() -> Cdouble

Wrapper for C++ function: `log`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `log`
- Type safety:  From compilation
"""

function log()::Cdouble
    ccall((:log, LIBRARY_PATH), Cdouble, (), )
end

"""
    trunc() -> Cdouble

Wrapper for C++ function: `trunc`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `trunc`
- Type safety:  From compilation
"""

function trunc()::Cdouble
    ccall((:trunc, LIBRARY_PATH), Cdouble, (), )
end

"""
    acos() -> Cdouble

Wrapper for C++ function: `acos`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `acos`
- Type safety:  From compilation
"""

function acos()::Cdouble
    ccall((:acos, LIBRARY_PATH), Cdouble, (), )
end

"""
    scalbnl() -> Float64

Wrapper for C++ function: `scalbnl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `scalbnl`
- Type safety:  From compilation
"""

function scalbnl()::Float64
    ccall((:scalbnl, LIBRARY_PATH), Float64, (), )
end

"""
    asinhl() -> Float64

Wrapper for C++ function: `asinhl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `asinhl`
- Type safety:  From compilation
"""

function asinhl()::Float64
    ccall((:asinhl, LIBRARY_PATH), Float64, (), )
end

"""
    powf() -> Cfloat

Wrapper for C++ function: `powf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `powf`
- Type safety:  From compilation
"""

function powf()::Cfloat
    ccall((:powf, LIBRARY_PATH), Cfloat, (), )
end

"""
    fmodf() -> Cfloat

Wrapper for C++ function: `fmodf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `fmodf`
- Type safety:  From compilation
"""

function fmodf()::Cfloat
    ccall((:fmodf, LIBRARY_PATH), Cfloat, (), )
end

"""
    remainderl() -> Float64

Wrapper for C++ function: `remainderl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `remainderl`
- Type safety:  From compilation
"""

function remainderl()::Float64
    ccall((:remainderl, LIBRARY_PATH), Float64, (), )
end

"""
    abs() -> Cint

Wrapper for C++ function: `abs`

# Arguments


# Returns
- `Cint`

# Metadata
- Mangled symbol: `abs`
- Type safety:  From compilation
"""

function abs()::Cint
    ccall((:abs, LIBRARY_PATH), Cint, (), )
end

"""
    log1pl() -> Float64

Wrapper for C++ function: `log1pl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `log1pl`
- Type safety:  From compilation
"""

function log1pl()::Float64
    ccall((:log1pl, LIBRARY_PATH), Float64, (), )
end

"""
    fdiml() -> Float64

Wrapper for C++ function: `fdiml`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `fdiml`
- Type safety:  From compilation
"""

function fdiml()::Float64
    ccall((:fdiml, LIBRARY_PATH), Float64, (), )
end

"""
    modf() -> Cdouble

Wrapper for C++ function: `modf`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `modf`
- Type safety:  From compilation
"""

function modf()::Cdouble
    ccall((:modf, LIBRARY_PATH), Cdouble, (), )
end

"""
    cos() -> Cdouble

Wrapper for C++ function: `cos`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `cos`
- Type safety:  From compilation
"""

function cos()::Cdouble
    ccall((:cos, LIBRARY_PATH), Cdouble, (), )
end

"""
    erfcl() -> Float64

Wrapper for C++ function: `erfcl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `erfcl`
- Type safety:  From compilation
"""

function erfcl()::Float64
    ccall((:erfcl, LIBRARY_PATH), Float64, (), )
end

"""
    remquof() -> Cfloat

Wrapper for C++ function: `remquof`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `remquof`
- Type safety:  From compilation
"""

function remquof()::Cfloat
    ccall((:remquof, LIBRARY_PATH), Cfloat, (), )
end

"""
    fma() -> Cdouble

Wrapper for C++ function: `fma`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `fma`
- Type safety:  From compilation
"""

function fma()::Cdouble
    ccall((:fma, LIBRARY_PATH), Cdouble, (), )
end

"""
    coshl() -> Float64

Wrapper for C++ function: `coshl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `coshl`
- Type safety:  From compilation
"""

function coshl()::Float64
    ccall((:coshl, LIBRARY_PATH), Float64, (), )
end

"""
    logl() -> Float64

Wrapper for C++ function: `logl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `logl`
- Type safety:  From compilation
"""

function logl()::Float64
    ccall((:logl, LIBRARY_PATH), Float64, (), )
end

"""
    tanhl() -> Float64

Wrapper for C++ function: `tanhl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `tanhl`
- Type safety:  From compilation
"""

function tanhl()::Float64
    ccall((:tanhl, LIBRARY_PATH), Float64, (), )
end

"""
    asinhf() -> Cfloat

Wrapper for C++ function: `asinhf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `asinhf`
- Type safety:  From compilation
"""

function asinhf()::Cfloat
    ccall((:asinhf, LIBRARY_PATH), Cfloat, (), )
end

"""
    _Z12check_status6Status() -> Any

Wrapper for C++ function: `_Z12check_status6Status`

# Arguments


# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z12check_status6Status`
- Type safety:  From compilation
"""

function _Z12check_status6Status()::Status
    return ccall((:_Z12check_status6Status, LIBRARY_PATH), Status, (), )
end

"""
    remainder() -> Cdouble

Wrapper for C++ function: `remainder`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `remainder`
- Type safety:  From compilation
"""

function remainder()::Cdouble
    ccall((:remainder, LIBRARY_PATH), Cdouble, (), )
end

"""
    lgammal() -> Float64

Wrapper for C++ function: `lgammal`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `lgammal`
- Type safety:  From compilation
"""

function lgammal()::Float64
    ccall((:lgammal, LIBRARY_PATH), Float64, (), )
end

"""
    log2() -> Cdouble

Wrapper for C++ function: `log2`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `log2`
- Type safety:  From compilation
"""

function log2()::Cdouble
    ccall((:log2, LIBRARY_PATH), Cdouble, (), )
end

"""
    lroundl() -> Clong

Wrapper for C++ function: `lroundl`

# Arguments


# Returns
- `Clong`

# Metadata
- Mangled symbol: `lroundl`
- Type safety:  From compilation
"""

function lroundl()::Clong
    ccall((:lroundl, LIBRARY_PATH), Clong, (), )
end

"""
    tanf() -> Cfloat

Wrapper for C++ function: `tanf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `tanf`
- Type safety:  From compilation
"""

function tanf()::Cfloat
    ccall((:tanf, LIBRARY_PATH), Cfloat, (), )
end

"""
    asinl() -> Float64

Wrapper for C++ function: `asinl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `asinl`
- Type safety:  From compilation
"""

function asinl()::Float64
    ccall((:asinl, LIBRARY_PATH), Float64, (), )
end

"""
    nextafterl() -> Float64

Wrapper for C++ function: `nextafterl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `nextafterl`
- Type safety:  From compilation
"""

function nextafterl()::Float64
    ccall((:nextafterl, LIBRARY_PATH), Float64, (), )
end

"""
    fmaxl() -> Float64

Wrapper for C++ function: `fmaxl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `fmaxl`
- Type safety:  From compilation
"""

function fmaxl()::Float64
    ccall((:fmaxl, LIBRARY_PATH), Float64, (), )
end

"""
    acosh() -> Cdouble

Wrapper for C++ function: `acosh`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `acosh`
- Type safety:  From compilation
"""

function acosh()::Cdouble
    ccall((:acosh, LIBRARY_PATH), Cdouble, (), )
end

"""
    llrint() -> Clonglong

Wrapper for C++ function: `llrint`

# Arguments


# Returns
- `Clonglong`

# Metadata
- Mangled symbol: `llrint`
- Type safety:  From compilation
"""

function llrint()::Clonglong
    ccall((:llrint, LIBRARY_PATH), Clonglong, (), )
end

"""
    scalbn() -> Cdouble

Wrapper for C++ function: `scalbn`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `scalbn`
- Type safety:  From compilation
"""

function scalbn()::Cdouble
    ccall((:scalbn, LIBRARY_PATH), Cdouble, (), )
end

"""
    exp2l() -> Float64

Wrapper for C++ function: `exp2l`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `exp2l`
- Type safety:  From compilation
"""

function exp2l()::Float64
    ccall((:exp2l, LIBRARY_PATH), Float64, (), )
end

"""
    frexpf() -> Cfloat

Wrapper for C++ function: `frexpf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `frexpf`
- Type safety:  From compilation
"""

function frexpf()::Cfloat
    ccall((:frexpf, LIBRARY_PATH), Cfloat, (), )
end

"""
    copysignl() -> Float64

Wrapper for C++ function: `copysignl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `copysignl`
- Type safety:  From compilation
"""

function copysignl()::Float64
    ccall((:copysignl, LIBRARY_PATH), Float64, (), )
end

"""
    rintf() -> Cfloat

Wrapper for C++ function: `rintf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `rintf`
- Type safety:  From compilation
"""

function rintf()::Cfloat
    ccall((:rintf, LIBRARY_PATH), Cfloat, (), )
end

"""
    roundf() -> Cfloat

Wrapper for C++ function: `roundf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `roundf`
- Type safety:  From compilation
"""

function roundf()::Cfloat
    ccall((:roundf, LIBRARY_PATH), Cfloat, (), )
end

"""
    log1pf() -> Cfloat

Wrapper for C++ function: `log1pf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `log1pf`
- Type safety:  From compilation
"""

function log1pf()::Cfloat
    ccall((:log1pf, LIBRARY_PATH), Cfloat, (), )
end

"""
    lgamma() -> Cdouble

Wrapper for C++ function: `lgamma`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `lgamma`
- Type safety:  From compilation
"""

function lgamma()::Cdouble
    ccall((:lgamma, LIBRARY_PATH), Cdouble, (), )
end

"""
    _Z9run_testsv() -> Cint

Wrapper for C++ function: `_Z9run_testsv`

# Arguments


# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z9run_testsv`
- Type safety:  From compilation
"""

function _Z9run_testsv()::Cint
    ccall((:_Z9run_testsv, LIBRARY_PATH), Cint, (), )
end

"""
    _Z17get_primary_colorv() -> Any

Wrapper for C++ function: `_Z17get_primary_colorv`

# Arguments


# Returns
- `Any`

# Metadata
- Mangled symbol: `_Z17get_primary_colorv`
- Type safety:  From compilation
"""

function _Z17get_primary_colorv()::Color
    return ccall((:_Z17get_primary_colorv, LIBRARY_PATH), Color, (), )
end

"""
    fminf() -> Cfloat

Wrapper for C++ function: `fminf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `fminf`
- Type safety:  From compilation
"""

function fminf()::Cfloat
    ccall((:fminf, LIBRARY_PATH), Cfloat, (), )
end

"""
    sinh() -> Cdouble

Wrapper for C++ function: `sinh`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `sinh`
- Type safety:  From compilation
"""

function sinh()::Cdouble
    ccall((:sinh, LIBRARY_PATH), Cdouble, (), )
end

"""
    ceil() -> Cdouble

Wrapper for C++ function: `ceil`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `ceil`
- Type safety:  From compilation
"""

function ceil()::Cdouble
    ccall((:ceil, LIBRARY_PATH), Cdouble, (), )
end

"""
    llroundf() -> Clonglong

Wrapper for C++ function: `llroundf`

# Arguments


# Returns
- `Clonglong`

# Metadata
- Mangled symbol: `llroundf`
- Type safety:  From compilation
"""

function llroundf()::Clonglong
    ccall((:llroundf, LIBRARY_PATH), Clonglong, (), )
end

"""
    expm1() -> Cdouble

Wrapper for C++ function: `expm1`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `expm1`
- Type safety:  From compilation
"""

function expm1()::Cdouble
    ccall((:expm1, LIBRARY_PATH), Cdouble, (), )
end

"""
    atanhf() -> Cfloat

Wrapper for C++ function: `atanhf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `atanhf`
- Type safety:  From compilation
"""

function atanhf()::Cfloat
    ccall((:atanhf, LIBRARY_PATH), Cfloat, (), )
end

"""
    llroundl() -> Clonglong

Wrapper for C++ function: `llroundl`

# Arguments


# Returns
- `Clonglong`

# Metadata
- Mangled symbol: `llroundl`
- Type safety:  From compilation
"""

function llroundl()::Clonglong
    ccall((:llroundl, LIBRARY_PATH), Clonglong, (), )
end

"""
    nearbyintl() -> Float64

Wrapper for C++ function: `nearbyintl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `nearbyintl`
- Type safety:  From compilation
"""

function nearbyintl()::Float64
    ccall((:nearbyintl, LIBRARY_PATH), Float64, (), )
end

"""
    tgamma() -> Cdouble

Wrapper for C++ function: `tgamma`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `tgamma`
- Type safety:  From compilation
"""

function tgamma()::Cdouble
    ccall((:tgamma, LIBRARY_PATH), Cdouble, (), )
end

"""
    cosl() -> Float64

Wrapper for C++ function: `cosl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `cosl`
- Type safety:  From compilation
"""

function cosl()::Float64
    ccall((:cosl, LIBRARY_PATH), Float64, (), )
end

"""
    expm1l() -> Float64

Wrapper for C++ function: `expm1l`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `expm1l`
- Type safety:  From compilation
"""

function expm1l()::Float64
    ccall((:expm1l, LIBRARY_PATH), Float64, (), )
end

"""
    ceill() -> Float64

Wrapper for C++ function: `ceill`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `ceill`
- Type safety:  From compilation
"""

function ceill()::Float64
    ccall((:ceill, LIBRARY_PATH), Float64, (), )
end

"""
    exp2f() -> Cfloat

Wrapper for C++ function: `exp2f`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `exp2f`
- Type safety:  From compilation
"""

function exp2f()::Cfloat
    ccall((:exp2f, LIBRARY_PATH), Cfloat, (), )
end

"""
    sinhf() -> Cfloat

Wrapper for C++ function: `sinhf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `sinhf`
- Type safety:  From compilation
"""

function sinhf()::Cfloat
    ccall((:sinhf, LIBRARY_PATH), Cfloat, (), )
end

"""
    expl() -> Float64

Wrapper for C++ function: `expl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `expl`
- Type safety:  From compilation
"""

function expl()::Float64
    ccall((:expl, LIBRARY_PATH), Float64, (), )
end

"""
    tan() -> Cdouble

Wrapper for C++ function: `tan`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `tan`
- Type safety:  From compilation
"""

function tan()::Cdouble
    ccall((:tan, LIBRARY_PATH), Cdouble, (), )
end

"""
    rintl() -> Float64

Wrapper for C++ function: `rintl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `rintl`
- Type safety:  From compilation
"""

function rintl()::Float64
    ccall((:rintl, LIBRARY_PATH), Float64, (), )
end

"""
    floorl() -> Float64

Wrapper for C++ function: `floorl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `floorl`
- Type safety:  From compilation
"""

function floorl()::Float64
    ccall((:floorl, LIBRARY_PATH), Float64, (), )
end

"""
    ldexpf() -> Cfloat

Wrapper for C++ function: `ldexpf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `ldexpf`
- Type safety:  From compilation
"""

function ldexpf()::Cfloat
    ccall((:ldexpf, LIBRARY_PATH), Cfloat, (), )
end

"""
    frexp() -> Cdouble

Wrapper for C++ function: `frexp`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `frexp`
- Type safety:  From compilation
"""

function frexp()::Cdouble
    ccall((:frexp, LIBRARY_PATH), Cdouble, (), )
end

"""
    nanf() -> Cfloat

Wrapper for C++ function: `nanf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `nanf`
- Type safety:  From compilation
"""

function nanf()::Cfloat
    ccall((:nanf, LIBRARY_PATH), Cfloat, (), )
end

"""
    rint() -> Cdouble

Wrapper for C++ function: `rint`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `rint`
- Type safety:  From compilation
"""

function rint()::Cdouble
    ccall((:rint, LIBRARY_PATH), Cdouble, (), )
end

"""
    atanf() -> Cfloat

Wrapper for C++ function: `atanf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `atanf`
- Type safety:  From compilation
"""

function atanf()::Cfloat
    ccall((:atanf, LIBRARY_PATH), Cfloat, (), )
end

"""
    _Z14apply_callbackPFiddEdd() -> Cint

Wrapper for C++ function: `_Z14apply_callbackPFiddEdd`

# Arguments


# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z14apply_callbackPFiddEdd`
- Type safety:  From compilation
"""

function _Z14apply_callbackPFiddEdd()::Cint
    ccall((:_Z14apply_callbackPFiddEdd, LIBRARY_PATH), Cint, (), )
end

"""
    logf() -> Cfloat

Wrapper for C++ function: `logf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `logf`
- Type safety:  From compilation
"""

function logf()::Cfloat
    ccall((:logf, LIBRARY_PATH), Cfloat, (), )
end

"""
    ilogb() -> Cint

Wrapper for C++ function: `ilogb`

# Arguments


# Returns
- `Cint`

# Metadata
- Mangled symbol: `ilogb`
- Type safety:  From compilation
"""

function ilogb()::Cint
    ccall((:ilogb, LIBRARY_PATH), Cint, (), )
end

"""
    hypotf() -> Cfloat

Wrapper for C++ function: `hypotf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `hypotf`
- Type safety:  From compilation
"""

function hypotf()::Cfloat
    ccall((:hypotf, LIBRARY_PATH), Cfloat, (), )
end

"""
    nan() -> Cdouble

Wrapper for C++ function: `nan`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `nan`
- Type safety:  From compilation
"""

function nan()::Cdouble
    ccall((:nan, LIBRARY_PATH), Cdouble, (), )
end

"""
    fmax() -> Cdouble

Wrapper for C++ function: `fmax`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `fmax`
- Type safety:  From compilation
"""

function fmax()::Cdouble
    ccall((:fmax, LIBRARY_PATH), Cdouble, (), )
end

"""
    acoshf() -> Cfloat

Wrapper for C++ function: `acoshf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `acoshf`
- Type safety:  From compilation
"""

function acoshf()::Cfloat
    ccall((:acoshf, LIBRARY_PATH), Cfloat, (), )
end

"""
    atan2f() -> Cfloat

Wrapper for C++ function: `atan2f`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `atan2f`
- Type safety:  From compilation
"""

function atan2f()::Cfloat
    ccall((:atan2f, LIBRARY_PATH), Cfloat, (), )
end

"""
    nanl() -> Float64

Wrapper for C++ function: `nanl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `nanl`
- Type safety:  From compilation
"""

function nanl()::Float64
    ccall((:nanl, LIBRARY_PATH), Float64, (), )
end

"""
    ldexp() -> Cdouble

Wrapper for C++ function: `ldexp`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `ldexp`
- Type safety:  From compilation
"""

function ldexp()::Cdouble
    ccall((:ldexp, LIBRARY_PATH), Cdouble, (), )
end

"""
    atanl() -> Float64

Wrapper for C++ function: `atanl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `atanl`
- Type safety:  From compilation
"""

function atanl()::Float64
    ccall((:atanl, LIBRARY_PATH), Float64, (), )
end

"""
    truncf() -> Cfloat

Wrapper for C++ function: `truncf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `truncf`
- Type safety:  From compilation
"""

function truncf()::Cfloat
    ccall((:truncf, LIBRARY_PATH), Cfloat, (), )
end

"""
    exp2() -> Cdouble

Wrapper for C++ function: `exp2`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `exp2`
- Type safety:  From compilation
"""

function exp2()::Cdouble
    ccall((:exp2, LIBRARY_PATH), Cdouble, (), )
end

"""
    atan() -> Cdouble

Wrapper for C++ function: `atan`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `atan`
- Type safety:  From compilation
"""

function atan()::Cdouble
    ccall((:atan, LIBRARY_PATH), Cdouble, (), )
end

"""
    _Z12color_to_int5Color() -> Cint

Wrapper for C++ function: `_Z12color_to_int5Color`

# Arguments


# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z12color_to_int5Color`
- Type safety:  From compilation
"""

function _Z12color_to_int5Color()::Cint
    ccall((:_Z12color_to_int5Color, LIBRARY_PATH), Cint, (), )
end

"""
    fmaxf() -> Cfloat

Wrapper for C++ function: `fmaxf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `fmaxf`
- Type safety:  From compilation
"""

function fmaxf()::Cfloat
    ccall((:fmaxf, LIBRARY_PATH), Cfloat, (), )
end

"""
    nextafter() -> Cdouble

Wrapper for C++ function: `nextafter`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `nextafter`
- Type safety:  From compilation
"""

function nextafter()::Cdouble
    ccall((:nextafter, LIBRARY_PATH), Cdouble, (), )
end

"""
    nextafterf() -> Cfloat

Wrapper for C++ function: `nextafterf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `nextafterf`
- Type safety:  From compilation
"""

function nextafterf()::Cfloat
    ccall((:nextafterf, LIBRARY_PATH), Cfloat, (), )
end

"""
    remquol() -> Float64

Wrapper for C++ function: `remquol`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `remquol`
- Type safety:  From compilation
"""

function remquol()::Float64
    ccall((:remquol, LIBRARY_PATH), Float64, (), )
end

"""
    sinhl() -> Float64

Wrapper for C++ function: `sinhl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `sinhl`
- Type safety:  From compilation
"""

function sinhl()::Float64
    ccall((:sinhl, LIBRARY_PATH), Float64, (), )
end

"""
    _Z22create_identity_matrixv() -> Matrix3x3

Wrapper for C++ function: `_Z22create_identity_matrixv`

# Arguments


# Returns
- `Matrix3x3`

# Metadata
- Mangled symbol: `_Z22create_identity_matrixv`
- Type safety:  From compilation
"""

function _Z22create_identity_matrixv()::Matrix3x3
    ccall((:_Z22create_identity_matrixv, LIBRARY_PATH), Matrix3x3, (), )
end

"""
    tanh() -> Cdouble

Wrapper for C++ function: `tanh`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `tanh`
- Type safety:  From compilation
"""

function tanh()::Cdouble
    ccall((:tanh, LIBRARY_PATH), Cdouble, (), )
end

"""
    log10l() -> Float64

Wrapper for C++ function: `log10l`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `log10l`
- Type safety:  From compilation
"""

function log10l()::Float64
    ccall((:log10l, LIBRARY_PATH), Float64, (), )
end

"""
    _Z8grid_get4Gridii() -> Cint

Wrapper for C++ function: `_Z8grid_get4Gridii`

# Arguments


# Returns
- `Cint`

# Metadata
- Mangled symbol: `_Z8grid_get4Gridii`
- Type safety:  From compilation
"""

function _Z8grid_get4Gridii()::Cint
    ccall((:_Z8grid_get4Gridii, LIBRARY_PATH), Cint, (), )
end

"""
    scalblnl() -> Float64

Wrapper for C++ function: `scalblnl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `scalblnl`
- Type safety:  From compilation
"""

function scalblnl()::Float64
    ccall((:scalblnl, LIBRARY_PATH), Float64, (), )
end

"""
    fmodl() -> Float64

Wrapper for C++ function: `fmodl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `fmodl`
- Type safety:  From compilation
"""

function fmodl()::Float64
    ccall((:fmodl, LIBRARY_PATH), Float64, (), )
end

"""
    log10() -> Cdouble

Wrapper for C++ function: `log10`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `log10`
- Type safety:  From compilation
"""

function log10()::Cdouble
    ccall((:log10, LIBRARY_PATH), Cdouble, (), )
end

"""
    logbl() -> Float64

Wrapper for C++ function: `logbl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `logbl`
- Type safety:  From compilation
"""

function logbl()::Float64
    ccall((:logbl, LIBRARY_PATH), Float64, (), )
end

"""
    erf() -> Cdouble

Wrapper for C++ function: `erf`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `erf`
- Type safety:  From compilation
"""

function erf()::Cdouble
    ccall((:erf, LIBRARY_PATH), Cdouble, (), )
end

"""
    fabsl() -> Float64

Wrapper for C++ function: `fabsl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `fabsl`
- Type safety:  From compilation
"""

function fabsl()::Float64
    ccall((:fabsl, LIBRARY_PATH), Float64, (), )
end

"""
    modff() -> Cfloat

Wrapper for C++ function: `modff`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `modff`
- Type safety:  From compilation
"""

function modff()::Cfloat
    ccall((:modff, LIBRARY_PATH), Cfloat, (), )
end

"""
    sqrtf() -> Cfloat

Wrapper for C++ function: `sqrtf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `sqrtf`
- Type safety:  From compilation
"""

function sqrtf()::Cfloat
    ccall((:sqrtf, LIBRARY_PATH), Cfloat, (), )
end

"""
    scalbnf() -> Cfloat

Wrapper for C++ function: `scalbnf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `scalbnf`
- Type safety:  From compilation
"""

function scalbnf()::Cfloat
    ccall((:scalbnf, LIBRARY_PATH), Cfloat, (), )
end

"""
    logb() -> Cdouble

Wrapper for C++ function: `logb`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `logb`
- Type safety:  From compilation
"""

function logb()::Cdouble
    ccall((:logb, LIBRARY_PATH), Cdouble, (), )
end

"""
    tgammaf() -> Cfloat

Wrapper for C++ function: `tgammaf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `tgammaf`
- Type safety:  From compilation
"""

function tgammaf()::Cfloat
    ccall((:tgammaf, LIBRARY_PATH), Cfloat, (), )
end

"""
    sqrtl() -> Float64

Wrapper for C++ function: `sqrtl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `sqrtl`
- Type safety:  From compilation
"""

function sqrtl()::Float64
    ccall((:sqrtl, LIBRARY_PATH), Float64, (), )
end

"""
    sinl() -> Float64

Wrapper for C++ function: `sinl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `sinl`
- Type safety:  From compilation
"""

function sinl()::Float64
    ccall((:sinl, LIBRARY_PATH), Float64, (), )
end

"""
    logbf() -> Cfloat

Wrapper for C++ function: `logbf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `logbf`
- Type safety:  From compilation
"""

function logbf()::Cfloat
    ccall((:logbf, LIBRARY_PATH), Cfloat, (), )
end

"""
    modfl() -> Float64

Wrapper for C++ function: `modfl`

# Arguments


# Returns
- `Float64`

# Metadata
- Mangled symbol: `modfl`
- Type safety:  From compilation
"""

function modfl()::Float64
    ccall((:modfl, LIBRARY_PATH), Float64, (), )
end

"""
    _Z14create_complex5Color6StatusPFiddE() -> ComplexType

Wrapper for C++ function: `_Z14create_complex5Color6StatusPFiddE`

# Arguments


# Returns
- `ComplexType`

# Metadata
- Mangled symbol: `_Z14create_complex5Color6StatusPFiddE`
- Type safety:  From compilation
"""

function _Z14create_complex5Color6StatusPFiddE()::ComplexType
    ccall((:_Z14create_complex5Color6StatusPFiddE, LIBRARY_PATH), ComplexType, (), )
end

"""
    fdimf() -> Cfloat

Wrapper for C++ function: `fdimf`

# Arguments


# Returns
- `Cfloat`

# Metadata
- Mangled symbol: `fdimf`
- Type safety:  From compilation
"""

function fdimf()::Cfloat
    ccall((:fdimf, LIBRARY_PATH), Cfloat, (), )
end

"""
    sin() -> Cdouble

Wrapper for C++ function: `sin`

# Arguments


# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `sin`
- Type safety:  From compilation
"""

function sin()::Cdouble
    ccall((:sin, LIBRARY_PATH), Cdouble, (), )
end


end # module TestAdvanced
