# test_libcxx_exports.jl — the STL path binds only what the library exports, and
# libc++'s own visibility does not get to decide what that is.
#
# libc++ 23 marks every function it does not annotate explicitly as hidden, with
# a `#pragma clang attribute` that applies on COFF too. On MinGW clang turns hidden
# into `-exclude-symbols:`, so std::vector<int>'s members were `T` in nm and absent
# from the DLL's export directory. The STL thunks could not bind, the JIT module
# failed, and every Tier-2 call into the library died with it (stl_test, LLVM
# 23.1.3, Windows: 222 exports, all seven STL testsets erroring). Two guards:
# create_library makes hidden libc++ definitions exportable for packages with
# [types] templates, and the STL method table keeps only symbols the loader can
# bind. No toolchain.

using Test
using RepliBuild

const _LX_C  = RepliBuild.Compiler
const _LX_CM = RepliBuild.ConfigurationManager

function _lx_config(; language::String, templates::Vector{String})
    dir = mktempdir()
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "lxprobe"
    root = "."
    [wrap]
    language = "$language"
    [types]
    templates = [$(join(repr.(templates), ", "))]
    """)
    return _LX_CM.load_config(toml)
end

# One line per case; `=>` gives the expected rewrite, `nothing` means unchanged.
const _LX_CASES = [
    # libc++ members hidden by the pragma: un-hidden
    "define linkonce_odr hidden void @_ZNSt3__16vectorIiNS_9allocatorIiEEEC2B9nqe230103Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #1 comdat align 2 !dbg !10 {" =>
        "define linkonce_odr void @_ZNSt3__16vectorIiNS_9allocatorIiEEEC2B9nqe230103Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #1 comdat align 2 !dbg !10 {",
    # const member (`NK`), return attributes before the type
    "define linkonce_odr hidden noundef i64 @_ZNKSt3__16vectorIiNS_9allocatorIiEEE4sizeB9nqe230103Ev(ptr %0) #1 comdat {" =>
        "define linkonce_odr noundef i64 @_ZNKSt3__16vectorIiNS_9allocatorIiEEE4sizeB9nqe230103Ev(ptr %0) #1 comdat {",
    # rvalue-ref-qualified member (`NO`), a literal struct return type with spaces
    "define weak_odr hidden { i64, i64 } @_ZNOSt3__14pairIiiE4takeEv(ptr %0) {" =>
        "define weak_odr { i64, i64 } @_ZNOSt3__14pairIiiE4takeEv(ptr %0) {",
    # ABI v2 namespace, no linkage keyword, a calling convention
    "define hidden fastcc void @_ZNSt3__24listIiE5clearEv(ptr %0) {" =>
        "define fastcc void @_ZNSt3__24listIiE5clearEv(ptr %0) {",
    # left alone: already default visibility
    "define linkonce_odr dso_local void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC2B9nqe230103Ev(ptr %0) {" => nothing,
    # left alone: the package's own -fvisibility=hidden
    "define hidden void @_Z8internalv() {" => nothing,
    # left alone: clang's helper and a lambda local to a libc++ function (`_ZZ`)
    "define linkonce_odr hidden void @__clang_call_terminate(ptr %0) {" => nothing,
    "define linkonce_odr hidden void @_ZZNSt3__16vectorIiNS_9allocatorIiEEE12emplace_backEiENKUlvE_clEv(ptr %0) {" => nothing,
    # left alone: libstdc++ (`std::__cxx11` is not libc++'s versioned namespace)
    "define linkonce_odr hidden void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5clearEv(ptr %0) {" => nothing,
    # left alone: declarations and uses are not definitions
    "declare hidden void @_ZNSt3__16vectorIiNS_9allocatorIiEEE9push_backEOi(ptr, ptr)" => nothing,
    "  call void @_ZNSt3__16vectorIiNS_9allocatorIiEEEC2B9nqe230103Ev(ptr %v) ; hidden" => nothing,
]

@testset "libc++ members the STL path binds are exported" begin
    @testset "only hidden std::__N definitions are un-hidden" begin
        dir = mktempdir()
        src = joinpath(dir, "mod.ll")
        write(src, "; ModuleID = 'probe'\n" * join(first.(_LX_CASES), "\n") * "\n")
        out_dir = joinpath(dir, "out")

        (files, n) = _LX_C._unhide_libcxx_ir([src], out_dir)
        @test n == count(p -> last(p) !== nothing, _LX_CASES)
        @test only(files) == joinpath(out_dir, "mod.ll")
        got = readlines(only(files))
        @test got[1] == "; ModuleID = 'probe'"            # untouched lines survive
        checked = 0
        for (i, (line, want)) in enumerate(_LX_CASES)
            @test got[i + 1] == something(want, line)
            checked += 1
        end
        @test checked == length(_LX_CASES)
        @test read(src, String) == "; ModuleID = 'probe'\n" * join(first.(_LX_CASES), "\n") * "\n"

        # Reused while newer than its input: the per-TU fallback keeps its object cache.
        (again, n2) = _LX_C._unhide_libcxx_ir([src], out_dir)
        @test again == files && n2 == 0
    end

    @testset "a module with nothing to change is not copied" begin
        dir = mktempdir()
        src = joinpath(dir, "plain.ll")
        write(src, "define hidden void @_Z8internalv() {\n  ret void\n}\n")
        out_dir = joinpath(dir, "out")
        (files, n) = _LX_C._unhide_libcxx_ir([src], out_dir)
        @test files == [src] && n == 0
        @test !isdir(out_dir)
    end

    @testset "only C++ packages with [types] templates are rewritten" begin
        hidden = "define linkonce_odr hidden void @_ZNSt3__16vectorIiNS_9allocatorIiEEEC2Ev(ptr %0) {\n"
        cases = [(_lx_config(language="c",   templates=["std::vector<int>"]), false),
                 (_lx_config(language="cpp", templates=String[]),             false),
                 (_lx_config(language="cpp", templates=["std::vector<int>"]), true)]
        for (cfg, rewritten) in cases
            src = joinpath(mktempdir(), "linked.ll")
            write(src, hidden)
            out = redirect_stdout(devnull) do
                _LX_C._stl_exportable_ir(cfg, [src])
            end
            want = rewritten ? joinpath(_LX_CM.get_build_path(cfg), "libcxx_exported", "linked.ll") : src
            @test only(out) == want
            @test occursin(" hidden ", read(only(out), String)) == !rewritten
        end
    end

    # Both IR -> machine code sites in create_library: the merged module and the
    # per-TU fallback. Read from the source, the way test_codegen_flags does.
    @testset "create_library rewrites before both codegen paths" begin
        src = read(joinpath(pkgdir(RepliBuild), "src", "Builder", "Compiler.jl"), String)
        @test occursin("files = _stl_exportable_ir(config, files)", src)
        @test occursin("split_inputs = _stl_exportable_ir(config, ", src)
    end

    # The table the STL thunks and factories are generated from. On PE, nm reads
    # the COFF symbol table, where an unexported member is still `T`.
    @testset "STL methods are gated to the runtime symbols" begin
        V = "_ZNSt3__16vectorIiNS_9allocatorIiEEE"
        mangled = """
        0000000180001000 T $(V)C1B9nqe230103Ev
        0000000180001010 T $(V)C2B9nqe230103Ev
        0000000180001020 T $(V)9push_backB9nqe230103EOi
        0000000180001030 T _ZNK$(V[4:end])4sizeB9nqe230103Ev
        """
        demangled = """
        0000000180001000 T std::__1::vector<int, std::__1::allocator<int>>::vector[abi:nqe230103]()
        0000000180001010 T std::__1::vector<int, std::__1::allocator<int>>::vector[abi:nqe230103]()
        0000000180001020 T std::__1::vector<int, std::__1::allocator<int>>::push_back[abi:nqe230103](int&&)
        0000000180001030 T std::__1::vector<int, std::__1::allocator<int>>::size[abi:nqe230103]() const
        """
        tmpl = ["std::vector<int>"]
        methods(m) = Dict(x["method"] => x["mangled"] for x in m["std::vector<int>"])

        # No runtime set: nm's answer, as before.
        (m0, u0) = _LX_C._stl_methods_from_nm(mangled, demangled, tmpl; runtime=nothing)
        @test methods(m0) == Dict("constructor" => "$(V)C1B9nqe230103Ev",
                                  "push_back"   => "$(V)9push_backB9nqe230103EOi",
                                  "size"        => "_ZNK$(V[4:end])4sizeB9nqe230103Ev")
        @test isempty(u0)

        # Only C2 and size are exported. C1 comes first in nm and must not shadow
        # C2; push_back has no bindable variant and is reported.
        rt = Set(["$(V)C2B9nqe230103Ev", "_ZNK$(V[4:end])4sizeB9nqe230103Ev"])
        (m1, u1) = _LX_C._stl_methods_from_nm(mangled, demangled, tmpl; runtime=rt)
        @test methods(m1) == Dict("constructor" => "$(V)C2B9nqe230103Ev",
                                  "size"        => "_ZNK$(V[4:end])4sizeB9nqe230103Ev")
        @test u1 == ["std::vector<int>::push_back"]

        # Nothing exported: nothing generated, everything reported.
        (m2, u2) = _LX_C._stl_methods_from_nm(mangled, demangled, tmpl; runtime=Set{String}())
        @test isempty(m2)
        @test u2 == ["std::vector<int>::constructor", "std::vector<int>::push_back",
                     "std::vector<int>::size"]
    end
end
