# =============================================================================
# What a Tier 2 call accepts where C++ takes a pointer, live (2026-11-17)
#
# A pointer parameter and every `this` of a C++ wrapper is `::Any`, and
# `JITManager.invoke` boxed whatever arrived: a struct value passed as `this`,
# a Julia function where `int (*)(int)` goes, a `Vector` for a
# `const std::vector<int>&`, a `Ref{String}` for a `std::string&` were each
# read as an address (SIGSEGV), and an `Int16` given to an overload set fell
# into its `const char*` member. A record held as its bytes (a bitfield
# record) passed as `this` crashed too, where it should work. A `std::vector`
# by value crashed with the very value the library returned for one, because
# the thunk takes such a parameter by address. And a `const char*` took
# `pointer(s)`: a `SubString` was read past its end and an embedded NUL cut
# the string short.
#
# The body of a Tier 2 wrapper now converts every pointer argument with
# `JITManager.pointer_arg(<declared type>, x)` (`_pointer_arg_type` names the
# type), which refuses what is not a pointer to it before any C++ runs.
#
# This builds test/pointer_arg_test with the system compiler, wraps it, and
# calls each case from a child. Needs clang++ and libJLCS.
#
# Negative checks (each made, seen to fail, restored). With the conversion
# left out of the bodies (`_tier2_invoke_args` ignoring `pointer_types`: the
# old path) the child segfaults in the second case, the bitfield record as
# `this`. `nothing` from `_pointer_arg_type` for a container by value: it
# segfaults in `container_by_value_takes_the_returned_value`. `pointer(x)` for
# a C string, or `Ptr{UInt8}` for a `char*`: `c_strings` reads past the
# `SubString` and takes the embedded NUL. `_plain_data(T) = true`, an object
# nothing emits spelled `void*`, a function pointer spelled `void*`, a value
# with no conversion boxed as `invoke` boxed it: a segfault in the refusal
# each guards. `_array_matches` returning `true`, `_is_byte_record(T) =
# false`, no `Ptr{Cvoid}` conversion for `Managed<X>`: the case that names it
# fails.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const POINTER_ARG_DIR = joinpath(@__DIR__, "pointer_arg_test")

if requires("pointer arguments of a Tier 2 call, live", :libJLCS, :clangxx)
@testset "Pointer arguments of a Tier 2 call, live" begin
    toml_path = joinpath(POINTER_ARG_DIR, "replibuild.toml")
    src = joinpath(POINTER_ARG_DIR, "pointer_arg.cpp")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "pointerarg"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(POINTER_ARG_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC", "-O1"]
    source_files = ["$(escape_string(src))"]
    include_dirs = ["$(escape_string(POINTER_ARG_DIR))"]

    [link]
    enable_lto = false
    optimization_level = "1"

    [binary]
    type = "shared"

    [wrap]
    language = "cpp"

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
    wrapper = joinpath(POINTER_ARG_DIR, "julia", "Pointerarg.jl")
    @test isfile(wrapper)
    code = isfile(wrapper) ? read(wrapper, String) : ""

    @testset "the wrapper says what each pointer position is" begin
        body(name) = (m = match(Regex("^function " * name * "\\([^\\n]*\\n(.*?)^end", "ms"), code);
                      m === nothing ? "no function $name" : String(m.captures[1]))
        pa(t, x) = "RepliBuild.JITManager.pointer_arg($t, $x)"
        jm = "RepliBuild.JITManager"
        @test occursin(pa("Ptr{Vec3}", "this"), body("Vec3_length"))
        @test occursin(pa("Ptr{Vec3}", "this") * ", " * pa("Ptr{Vec3}", "o"), body("Vec3_dot"))
        @test occursin(pa("Ptr{Flags}", "this"), body("Flags_get_level"))
        @test occursin(pa("Ptr{$jm.Code}", "f") * ", Cint(x)", body("apply_unary"))
        @test occursin(pa("Ptr{Cvoid}", "p"), body("echo_ptr"))
        @test occursin(pa("Ptr{Cint}", "p"), body("sum_ints"))
        @test occursin(pa("Ptr{UInt8}", "p"), body("count_byte"))          # `unsigned char*` is data
        @test occursin(pa("Cstring", "s"), body("cstr_len"))               # `char*` is a C string
        @test occursin(pa("Cstring", "s"), body("overloaded_ptr"))
        @test occursin(pa("Ptr{$jm.Opaque}", "v"), body("vec_sum"))
        @test occursin(pa("$jm.ByValue{NTuple{24, UInt8}}", "v"), body("vec_sum_val"))
        @test occursin(pa("Ptr{$jm.Opaque}", "s") * ", " * pa("Cstring", "x"), body("str_append"))
        # A `std::string` the wrapper builds is not a pointer argument of the caller's.
        @test occursin("RepliBuild.StdStrings.string_arg(s, ", body("str_size_ref"))
        @test !occursin("pointer_arg", body("str_size_ref"))
        # An owning type converts to the untyped pointer a base's `this` takes.
        @test occursin("unsafe_convert(::Type{Ptr{Cvoid}}, obj::ManagedSquare) = Ptr{Cvoid}(obj.handle)", code)
        # Every `this` of a Tier 2 call is converted, never handed to `invoke` as it came.
        methods_ = [String(m.captures[2]) for m in eachmatch(r"^function (\w+)\(this::Any[^\n]*\n(.*?)^end"ms, code)]
        tier2 = filter(b -> occursin("JITManager.invoke", b), methods_)
        @test length(tier2) >= 8
        @test all(b -> occursin(r"pointer_arg\(Ptr\{[^,]+\}, this\)", b), tier2)
    end

    names = ["this_takes_ptr_ref_and_void_pointer", "byte_record_is_passed_by_its_bytes",
             "owning_objects_pass_as_their_pointer", "function_pointers",
             "void_pointer_takes_what_ccall_takes", "data_pointers_take_arrays_of_the_pointee",
             "container_by_value_takes_the_returned_value", "std_string_objects_behind_pointers",
             "c_strings", "overloads_still_dispatch",
             "string_is_not_a_void_pointer_or_an_object", "array_of_another_type_is_refused",
             "scalar_does_not_select_the_char_pointer_overload", "julia_function_is_not_a_function_pointer",
             "struct_value_is_not_this", "julia_vector_is_not_a_std_vector",
             "ref_to_a_julia_object_is_not_c_data"]
    @testset "every case, from a child" begin
        r = run_isolated(`$(joinpath(POINTER_ARG_DIR, "probe_pointer_arg.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end
end # requires
