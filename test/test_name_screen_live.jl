# =============================================================================
# The name screen, live (2026-11-17)
#
# test/name_screen_test holds two libraries whose identifiers are names Julia
# and a generated wrapper already use (audit 2026-11-17, B2, B3 and B5):
#
#   c/     functions `String`, `Ptr`, `Cint`, `eval`, `include`, `ccall`, `Base`,
#          `Libdl`, `RepliBuild`, `nothing`, `LIBRARY_PATH`, …; records `Integer`
#          and `Bool` beside C's own `bool`; an enumerator `Symbol`; a global
#          `Dict`
#   cpp/   records `Bool`, `Cvoid`, `Int`, `Integer`, `Cstring`, `UInt8`, made
#          and returned by value through thunks; a class whose methods `open`,
#          `isempty` and `String` become top-level proxies
#
# Before the screen (src/Wrapper/NameScreen.jl) the C wrap was refused for one
# function named `LIBRARY_PATH`; without that one the module failed to load on
# `eval`; without those, `String(1)` was 102 in every module of the process.
# The C++ module segfaulted or failed to load on the record names.
#
# This builds both with the real compilers, wraps them, and loads and calls
# both wrappers in one child (a load that fails, or a crash, takes the child):
# every library function is reachable under the name the wrap reports, and
# Base's types have no method from either module. Then the module's own name:
# a name the wrapper's text uses is an error that says what to set.
#
# Built in a temporary copy, so nothing is written under test/. Needs clang++
# and libJLCS for the C++ half.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const NAME_SCREEN_DIR = joinpath(@__DIR__, "name_screen_test")

# One of the fixture's two projects, in `root`. Returns the TOML's path.
function _name_screen_project(root, sub, name, language, flags, source; module_name = "")
    dir = joinpath(root, sub)
    isdir(dir) || cp(joinpath(NAME_SCREEN_DIR, sub), dir)
    toml = joinpath(dir, isempty(module_name) ? "replibuild.toml" : "replibuild_$(module_name).toml")
    write(toml, """
    [project]
    name = "$name"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(dir))"

    [compile]
    flags = $flags
    source_files = ["$(escape_string(joinpath(dir, source)))"]
    include_dirs = ["$(escape_string(dir))"]

    [link]
    enable_lto = false
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    language = "$language"
    $(isempty(module_name) ? "" : "module_name = \"$module_name\"")

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    return toml
end

# stdout of `f`, for the lines a wrap prints.
function _name_screen_output(f)
    path = tempname()
    open(path, "w") do io
        redirect_stdout(f, io)
    end
    text = read(path, String)
    rm(path; force = true)
    return text
end

if requires("the name screen, live", :libJLCS, :clangxx)
@testset "The name screen, live" begin
    root = mktempdir()
    c_toml = _name_screen_project(root, "c", "namescreenc", "c", "[\"-fPIC\"]", "names.c")
    cpp_toml = _name_screen_project(root, "cpp", "namescreencpp", "cpp", "[\"-std=c++17\", \"-fPIC\"]", "names.cpp")
    redirect_stdout(devnull) do
        RepliBuild.build(c_toml)
        RepliBuild.build(cpp_toml)
    end
    c_said = _name_screen_output(() -> RepliBuild.wrap(c_toml))
    cpp_said = _name_screen_output(() -> RepliBuild.wrap(cpp_toml))
    c_wrapper = joinpath(root, "c", "julia", "Namescreenc.jl")
    cpp_wrapper = joinpath(root, "cpp", "julia", "Namescreencpp.jl")
    @test isfile(c_wrapper) && isfile(cpp_wrapper)

    @testset "the wrap names each one, sorted, with the reason" begin
        reported(said) = [strip(l) for l in split(said, '\n') if occursin(" is bound as ", l)]
        c = reported(c_said)
        @test issorted(c) && !isempty(c)
        @test occursin("library name(s) bound under another name", c_said)
        for line in ("function `String` is bound as `String_`: `String` is a type of Base or Core, and a " *
                     "function of that name would be a method of it in every module",
                     "function `eval` is bound as `eval_`: every module binds its own `eval`",
                     "function `ccall` is bound as `ccall_`: Julia has no function named `ccall`",
                     "function `nothing` is bound as `nothing_`: the wrapper's own text uses `nothing`",
                     "function `LIBRARY_PATH` is bound as `LIBRARY_PATH_`: the wrapper itself binds `LIBRARY_PATH`",
                     "function `Namescreenc` is bound as `Namescreenc_`: the wrapper itself binds `Namescreenc`",
                     "type `Integer` is bound as `c_Integer`: the wrapper's own text uses `Integer`",
                     "enumerator `Symbol` is bound as `Kind_Symbol`: the wrapper's own text uses `Symbol`")
            @test line in c
        end
        # What the wrapper's text does not use is not in the report: `isfile`
        # is a function its header calls, as `Base.isfile`.
        @test !any(l -> occursin(r"`(length|print|show|isfile|Task|stat|plain)` is bound", l), c)
        x = reported(cpp_said)
        @test issorted(x)
        for n in ("Bool", "Cvoid", "Int", "Integer", "Cstring", "UInt8")
            @test "type `$n` is bound as `c_$n`: the wrapper's own text uses `$n`" in x
        end
        @test "function `String` is bound as `String_`: `String` is a type of Base or Core, and a " *
              "function of that name would be a method of it in every module" in x
        @test "function `Dict` is bound as `Dict_`: `Dict` is a type of Base or Core, and a " *
              "function of that name would be a method of it in every module" in x      # a global's accessor
        @test !any(l -> occursin(r"`(Task|Widget|get|size|open|isempty)` is bound", l), x)
        @test !any(l -> occursin("type `String`", l), x)       # the type sanitizer's own, as before
    end

    @testset "both wrappers, loaded and called from a child" begin
        r = run_isolated(`$(joinpath(NAME_SCREEN_DIR, "probe_names.jl")) $c_wrapper $cpp_wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["c_ordinary_functions", "c_moved_functions", "c_kept_functions",
                 "c_wrapper_names_are_its_own", "c_records_enumerators_globals",
                 "cpp_ordinary_functions", "cpp_records", "cpp_own_types_beside_the_records",
                 "cpp_record_holding_one_by_value", "cpp_method_proxies", "base_is_unchanged"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end

    @testset "the text: no name bound twice, the wrapper's names its own" begin
        W = RepliBuild.Wrapper
        for (wrapper, modname) in ((c_wrapper, "Namescreenc"), (cpp_wrapper, "Namescreencpp"))
            text = read(wrapper, String)
            W._assert_wrapper_loadable(text, modname)
            uses = W._name_uses(text)
            @test uses.bound[modname] == [:module]
            @test uses.bound["LIBRARY_PATH"] == [:const] && uses.bound["Libdl"] == [:import]
            # No library function is bound under a name of a type of Base.
            pirates = [n for (n, kinds) in uses.bound
                       if :function in kinds && !any(k -> k in (:struct, :type, :const), kinds) &&
                          W._function_name_refusal(n) === :base_type]
            @test pirates == String[]
        end
    end

    @testset "the module's own name: an error that says what to set" begin
        # The same library, with a module name the wrapper's text uses: the
        # name a project directory called `string` or `ptr` would get.
        for name in ("String", "Ptr")
            toml = _name_screen_project(root, "c", "namescreenc", "c", "[\"-fPIC\"]", "names.c"; module_name = name)
            err = try
                redirect_stdout(() -> RepliBuild.wrap(toml), devnull)
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("module would be named `$name`", err)
            @test occursin("`module_name` under `[wrap]`", err) && occursin(basename(toml), err)
            @test !isfile(joinpath(root, "c", "julia", "$name.jl"))
        end
        # A Base name the text does not use is a module name like any other.
        toml = _name_screen_project(root, "c", "namescreenc", "c", "[\"-fPIC\"]", "names.c"; module_name = "Task")
        redirect_stdout(() -> RepliBuild.wrap(toml), devnull)
        @test isfile(joinpath(root, "c", "julia", "Task.jl"))
    end
end
end # requires
