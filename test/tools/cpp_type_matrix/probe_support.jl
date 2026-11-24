# Shared by every family's probe.jl (the tool copies it beside the probe).
#
# A probe is a list of cases. A case is one cell of the matrix: it calls the
# wrapper the way a user would and answers with a verdict.
#
#     case("std::vector<int>", "pass by reference") do
#         v = M.create_std_vector_int(); push!(v, 2); push!(v, 5)
#         works(M.vec_sum_ref(v), 7)
#     end
#
# A probe uses what the manual gives a wrapper's user (`docs/src/calling.md`):
# the module's functions and types, `create_std_*()`, `Ref(x)` for a reference
# to a value the library returned. `fn(:name)` is the module's function for the
# library's `name`.
#
# Verdicts a case answers with:
#
#   works(got, want[, note])   WORKS, or WRONG with both values
#   opaque(got, want, why)     OPAQUE, or WRONG: the value crossed and the
#                              library reads it right, and Julia can neither
#                              read nor build what is inside
#   noway(why)                 NO WAY: bound, and nothing a wrapper gives can be
#                              passed, or the result cannot be used at all. It
#                              is the probe author's finding: say what was
#                              looked for
#   takes(f, want, what)       for "does it take a plain Julia value there":
#                              WORKS, or NOT TAKEN with the MethodError
#   nomake(r, why)             NO WAY for a "make" cell, after checking that the
#                              wrapper has no `create_…` function matching `r`
#   intact(control, got, want) for "is the value Julia holds still the object
#                              the library built": WORKS or WRONG, after the
#                              same check on one that never left C++
#
# An exception is classified here: a stated refusal is REFUSED, a missing
# function ABSENT, anything else ERROR.
#
# The five cells of the matrix are "make" (a value made from Julia, with no
# function of the library making it), "pass by value", "pass by reference",
# "return by value" and "return by reference". Any other cell name is listed
# under "other".
#
# The tool restarts a probe that dies in a call, from the case after it, so a
# case must not rely on an earlier one.
#
# Nothing is loaded here: the wrapper imports RepliBuild itself, and a probe
# that needs no wrapper (the tool's own test) starts in a second.

# How many cases an earlier run of this probe already started: the tool passes it.
const _TM_SKIP = something(tryparse(Int, get(ARGS, 1, "0")), 0)
const _TM_SEEN = Ref(0)
const _TM_MODULE = Ref{Module}()

"The wrapper module of the project this probe sits in."
function load_wrapper()
    files = filter(f -> endswith(f, ".jl"), readdir(joinpath(@__DIR__, "julia"); join = true))
    length(files) == 1 || error("expected one wrapper in julia/, found $(length(files))")
    _TM_MODULE[] = include(files[1])
    return _TM_MODULE[]
end

"""
The wrapper's function for the library's `name`. libstdc++ tags a function
whose return type holds a `std::string` (`[abi:cxx11]`), and the wrapper's name
carries the tag: `str_make` is bound as `str_make_abi_cxx11`.
"""
function fn(name::Symbol)
    M = _TM_MODULE[]
    isdefined(M, name) && return getfield(M, name)
    tagged = Symbol(name, :_abi_cxx11)
    isdefined(M, tagged) && return getfield(M, tagged)
    throw(UndefVarError(name))
end

_tm_short(s, n = 170) = (s = replace(strip(String(s)), r"\s+" => " "); length(s) > n ? first(s, n) * "…" : s)

# A type's name without the module the wrapper was loaded into (`Main.TmAny.any`).
_tm_typename(s::AbstractString) = replace(s, r"\bMain\.(?:\w+\.)?" => "")

_tm_method_error(e::MethodError) =
    "no method " * _tm_short(string(e.f) * "(" * join(("::" * _tm_typename(string(typeof(a))) for a in e.args), ", ") * ")", 150)

function _tm_classify(e)
    e isa UndefVarError && return "ABSENT `$(e.var)` is not in the wrapper"
    msg = sprint(showerror, e)
    if occursin("ABI Safety Trap", msg) || occursin(r"cannot (call|pass|return)", msg)
        return "REFUSED " * _tm_short(replace(msg, r"^.*?ABI Safety Trap:\s*"s => ""))
    end
    e isa MethodError && return "ERROR MethodError: " * _tm_method_error(e)
    return "ERROR " * _tm_short(first(split(msg, '\n')))
end

function case(f, type::AbstractString, cell::AbstractString)
    _TM_SEEN[] += 1
    _TM_SEEN[] <= _TM_SKIP && return
    # A line of its own, whatever the library printed before.
    println("\nPROBE_START ", type, " | ", cell); flush(stdout)
    v = try
        r = Base.invokelatest(f)
        r isa AbstractString ? String(r) : "ERROR the case answered with a $(typeof(r)), not a verdict"
    catch e
        _tm_classify(e)
    end
    println("\nPROBE ", type, " | ", cell, " : ", _tm_short(v, 400)); flush(stdout)
    return nothing
end

_tm_same(got, want) = (got isa AbstractFloat || want isa AbstractFloat) ? isapprox(got, want; rtol = 1e-12) : isequal(got, want)

_tm_wrong(got, want) = "WRONG got $(_tm_short(repr(got), 60)), the library's answer is $(_tm_short(repr(want), 60))"

"WORKS when `got` is what the library answers, else WRONG with both."
works(got, want, note::AbstractString = "") = _tm_same(got, want) ? rstrip("WORKS " * note) : _tm_wrong(got, want)

"OPAQUE when the library reads the value right and Julia cannot read or build what is inside; else WRONG."
opaque(got, want, why::AbstractString) = _tm_same(got, want) ? "OPAQUE " * why : _tm_wrong(got, want)

"Bound, and unusable from Julia: say what was looked for."
noway(why::AbstractString) = "NO WAY " * why

"""
Whether the wrapper takes a plain Julia value: WORKS when `f()` is `want`, NOT
TAKEN naming `what` when no method takes it. Any other failure is the case's.
"""
function takes(f, want, what::AbstractString)
    got = try
        f()
    catch e
        e isa MethodError || rethrow()
        return "NOT TAKEN " * what * ": " * _tm_method_error(e)
    end
    return works(got, want, what)
end

"The wrapper's `create_…` functions whose name matches `r`: what a user can make with no library function."
creators(r::Regex) = sort!(String[string(n) for n in names(_TM_MODULE[]; all = true)
                                  if startswith(string(n), "create_") && occursin(r, string(n))])

"NO WAY for a `make` cell, once the wrapper is seen to have no `create_…` function matching `r`."
function nomake(r::Regex, why::AbstractString)
    found = creators(r)
    isempty(found) || return "ERROR the wrapper has $(join(found, ", ")), which this probe does not try"
    return noway("no create function in the wrapper; " * why)
end

"""
A check the library makes on a value Julia holds (`got`), beside the same check
on a value that never left C++ (`control`): both must answer `want`. A check
that fails on its own control says nothing, and is the probe's ERROR.
"""
function intact(control, got, want)
    _tm_same(control, want) ||
        return "ERROR the probe's own check answers $(repr(control)) on a value that never left C++, not $(repr(want))"
    return _tm_same(got, want) ? "WORKS" :
        "WRONG the library's check answers $(repr(got)) on the value Julia holds, and $(repr(want)) on one that never left C++"
end

"What a call returned, for a verdict that has to describe it: its type."
describe(x) = _tm_short(_tm_typename(string(typeof(x))), 90)

done() = (println("\nPROBE_DONE"); flush(stdout))
