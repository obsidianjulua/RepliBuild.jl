# STL extract + force-TU regressions (library-free unit tests; optional -O2
# emission probe that needs clang++).
#
# Two defects found wrapping a host-STL Hub package at -O2 with no user TU
# that named `T t;`:
#
# 1. The force body took `T&` and never default-constructed, so `vector()` /
#    `map()` were absent from the .so. The factory memset-zeroed a live map.
# 2. `extract_stl_method_symbols` split `Class::method(args)` on the last `::`
#    at angle-bracket depth 0, so `std::allocator` in ctor args stole the
#    split and allocator/copy ctors never classified.
#
# The Hub package is not a fixture. These traces are the library-free
# reproductions: demangled strings for (2), generated C++ text + extra-flag
# helper for (1). The optional emission probe compiles the generated TU at
# -O2 through the same compile_single_to_ir path.

using Test
using RepliBuild

const C = RepliBuild.Compiler

@testset "STL class::method split is paren-aware" begin
    # The reproduction: last `::` is inside the allocator argument.
    demangled = "std::vector<int, std::allocator<int> >::vector(std::allocator<int> const&)"
    got = C._stl_split_class_method(demangled)
    @test got !== nothing
    class, sig = got
    @test class == "std::vector<int, std::allocator<int> >"
    @test sig == "vector(std::allocator<int> const&)"
    @test C._normalize_stl_type(class) == "std::vector<int>"

    # Empty-paren default ctor — the only spelling the old scan could see.
    d0 = "std::vector<int, std::allocator<int> >::vector()"
    class0, sig0 = C._stl_split_class_method(d0)
    @test class0 == "std::vector<int, std::allocator<int> >"
    @test sig0 == "vector()"

    # Methods with no `::` in the args still split at the class boundary.
    dsize = "std::vector<int, std::allocator<int> >::size() const"
    _, ssize = C._stl_split_class_method(dsize)
    @test ssize == "size() const"

    dpush = "std::vector<int, std::allocator<int> >::push_back(int&&)"
    _, spush = C._stl_split_class_method(dpush)
    @test spush == "push_back(int&&)"

    # map allocator ctor — several `std::` in the argument list.
    dmap = string(
        "std::map<int, int, std::less<int>, std::allocator<std::pair<int const, int> > >::",
        "map(std::allocator<std::pair<int const, int> > const&)")
    cmap, smap = C._stl_split_class_method(dmap)
    @test startswith(cmap, "std::map<int, int")
    @test startswith(smap, "map(")
    @test C._normalize_stl_type(cmap) == "std::map<int, int>"

    # string default ctor (inline namespace still on the class).
    ds = "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string()"
    cs, ss = C._stl_split_class_method(ds)
    @test ss == "basic_string()"
    @test C._normalize_stl_type(cs) == "std::basic_string<char>"

    @test C._stl_split_class_method("not_a_method") === nothing
end

@testset "STL default ctor is the only constructor extract binds" begin
    # Factory invokes T() with no extra args. Binding vector(allocator const&)
    # as "constructor" would be a one-arg call of a two-arg callee.
    @test C._classify_stl_method("vector()", "std::vector<int>") == ("constructor", false)
    @test C._classify_stl_method("vector( )", "std::vector<int>") == ("constructor", false)
    @test C._classify_stl_method("map()", "std::map<int, int>") == ("constructor", false)
    @test C._classify_stl_method("basic_string()", "std::basic_string<char>") == ("constructor", false)
    @test C._classify_stl_method("unordered_map()", "std::unordered_map<int, int>") == ("constructor", false)

    @test C._classify_stl_method("vector(std::allocator<int> const&)", "std::vector<int>") === nothing
    @test C._classify_stl_method("vector(unsigned long, std::allocator<int> const&)",
                                 "std::vector<int>") === nothing
    @test C._classify_stl_method("map(std::allocator<std::pair<int const, int> > const&)",
                                 "std::map<int, int>") === nothing

    # Methods still classify.
    @test C._classify_stl_method("size() const", "std::vector<int>") == ("size", true)
    @test C._classify_stl_method("~vector()", "std::vector<int>") == ("destructor", false)

    # Key erase, not iterator erase — CppMap.delete! passes Ptr{K}.
    @test C._classify_stl_method("erase(int const&)", "std::map<int, int>") == ("erase", false)
    @test C._classify_stl_method(
        "erase(std::_Rb_tree_iterator<std::pair<int const, int> >)",
        "std::map<int, int>") === nothing
    @test C._classify_stl_method(
        "erase(std::__detail::_Node_iterator<std::pair<int const, int>, false, false>)",
        "std::unordered_map<int, int>") === nothing
    # libc++ spells vector's iterator `__wrap_iter` — no "iterator" in the name,
    # so a `r"iterator"i` test alone is a guard that only holds on libstdc++.
    @test C._classify_stl_method(
        "erase(std::__1::__wrap_iter<int const*>)", "std::vector<int>") === nothing
    @test C._classify_stl_method(
        "erase(std::__1::__map_iterator<std::__1::__tree_iterator<int, void*, long long>>)",
        "std::map<int, int>") === nothing
    # ...and the key overload, the one CppMap.delete! actually calls, still binds
    # under either spelling.
    @test C._classify_stl_method("erase(int const&)", "std::map<int, int>") == ("erase", false)

    # Bucket begin/end(size_type) is not iterator begin/end().
    @test C._classify_stl_method("end() const", "std::map<int, int>") == ("end_", true)
    @test C._classify_stl_method("end(unsigned long) const",
                                 "std::unordered_map<int, int>") === nothing
    @test C._classify_stl_method("begin() const", "std::vector<int>") == ("begin", true)
    @test C._classify_stl_method("begin(unsigned long) const",
                                 "std::unordered_map<int, int>") === nothing
end

@testset "force TU default-constructs and the templates TU is -fno-inline" begin
    body = C._template_force_body("std::vector<int>", 1)
    @test occursin("void replibuild_force_1()", body)
    @test occursin("std::vector<int> c;", body)
    @test !occursin("std::vector<int>& c", body)

    @test C._is_template_instantiation_tu("/tmp/.replibuild_cache/replibuild_templates.cpp")
    @test !C._is_template_instantiation_tu("/tmp/stl.cpp")
    @test C._template_tu_compile_flags("replibuild_templates.cpp") == ["-fno-inline"]
    @test C._template_tu_compile_flags("stl.cpp") == String[]

    fp = C._file_compile_fingerprint("abc", "replibuild_templates.cpp")
    @test fp != "abc"
    @test C._file_compile_fingerprint("abc", "stl.cpp") == "abc"
    @test C._file_compile_fingerprint("", "replibuild_templates.cpp") == ""
end

# ── Optional: -O2 emission through the real compile path ─────────────────────
# Negative check from GENERATOR-stl-force-default-ctor.md: a templates-only
# package at -O2, no user `T t;`, must still have standalone vector() / map()
# and extract must bind them as constructor.

# `full_signature` keeps the raw demangled text, and on libc++ that carries the
# ABI tag `_classify_stl_method` strips before it decides anything. The tag is
# not part of the method's identity, so drop it before asserting which ctor bound.
_untag(sig::AbstractString) = replace(sig, r"\[abi:[^\]]*\]" => "")

function _clangxx_available()::Bool
    try
        p = run(pipeline(ignorestatus(`clang++ --version`), stdout=devnull, stderr=devnull))
        return p.exitcode == 0
    catch
        return false
    end
end

@testset "force TU at -O2 emits default ctors" begin
    if !_clangxx_available()
        @info "clang++ missing — skipping STL -O2 ctor emission probe"
        return
    end

    mktempdir() do dir
        toml = joinpath(dir, "replibuild.toml")
        write(toml, """
        [project]
        name = "stl_ctor_fix"
        root = "."

        [compile]
        flags = ["-O2", "-fPIC", "-std=c++17"]
        source_files = []

        [link]
        enable_lto = false
        optimization_level = "2"

        [binary]
        type = "shared"

        [wrap]
        language = "cpp"

        [types]
        templates = ["std::vector<int>", "std::map<int, int>"]
        template_headers = ["<vector>", "<map>"]

        [cache]
        enabled = false
        """)

        lib = RepliBuild.build(toml)
        @test lib !== nothing && isfile(lib)

        nm_out, nm_ec = RepliBuild.BuildBridge.execute("nm", ["-gC", "--defined-only", lib])
        @test nm_ec == 0
        # Spelled as an invariant, not as one standard library's answer: libc++
        # puts an inline namespace on the class (`std::__1::vector`) and mangles
        # a `_LIBCPP_HIDE_FROM_ABI` tag in after the method name
        # (`vector[abi:nqe220108]()`), so a libstdc++-shaped literal goes red on
        # Windows for a build that is in fact correct.
        @test occursin(r"std::(?:__\w+::)?vector<int.*::vector(?:\[abi:[^\]]*\])?\(\)", nm_out)
        @test occursin(r"std::(?:__\w+::)?map<int, int.*::map(?:\[abi:[^\]]*\])?\(\)", nm_out)
        @test occursin(r"std::(?:__\w+::)?map<int, int.*::~map(?:\[abi:[^\]]*\])?\(\)", nm_out)

        methods = C.extract_stl_method_symbols(lib, ["std::vector<int>", "std::map<int, int>"])
        @test haskey(methods, "std::vector<int>")
        @test haskey(methods, "std::map<int, int>")

        vctors = [m for m in methods["std::vector<int>"] if m["method"] == "constructor"]
        @test length(vctors) == 1
        @test _untag(vctors[1]["full_signature"]) == "vector()"

        mctors = [m for m in methods["std::map<int, int>"] if m["method"] == "constructor"]
        @test length(mctors) == 1
        @test _untag(mctors[1]["full_signature"]) == "map()"

        @test any(m -> m["method"] == "destructor", methods["std::map<int, int>"])
    end
end
