# test_jlcs_soname.jl — libJLCS's soname is the dialect's, and a thunks library
# linked against another one is refused.
#
# add_llvm_library gives a shared library LLVM's version as its soname, so
# libJLCS was `libJLCS.so.23.1` (`libJLCS.so.23.1jl` from LLVM_full_jll): the
# name followed the LLVM it was built with. A thunks library records the
# soname of the libJLCS it was linked against, and the loader satisfies that
# import from an already-loaded library only when the sonames are equal. When
# they are not it searches the thunks library's RUNPATH, finds the old file
# where it still is, and loads a second libJLCS without a word: two exception
# buffers, and a C++ exception a thunk catches never reaches Julia.
#
# So the soname is `libJLCS.so.<ABI_VERSION>`, set by the dialect's own CMake,
# and `JITManager.open_thunks_library` refuses a thunks library that brought
# another libJLCS into the process.
#
# No toolchain. test_jlcs_soname_live.jl reads a real library's soname and
# makes the loader do it.

using Test
using RepliBuild

const _SN_JM = RepliBuild.JITManager
const _SN_MLIR = joinpath(dirname(@__DIR__), "src", "mlir")

@testset "the soname is the dialect's ABI version" begin
    cmake = read(joinpath(_SN_MLIR, "CMakeLists.txt"), String)
    abi = parse(Int, strip(read(joinpath(_SN_MLIR, "ABI_VERSION"), String)))
    version = VersionNumber(strip(read(joinpath(_SN_MLIR, "JLCS_VERSION"), String)))

    # libJLCS.so.<ABI> is a link to libJLCS.so.<VERSION>: one major.
    @test version.major == abi
    @test occursin("file(STRINGS \"\${CMAKE_CURRENT_SOURCE_DIR}/JLCS_VERSION\" JLCS_VERSION", cmake)

    # Set on the target, AFTER add_llvm_library, which sets LLVM's: the later
    # call is the one that holds.
    at_library = findfirst("add_llvm_library(JLCS SHARED", cmake)
    at_soname = findfirst(r"set_target_properties\(JLCS PROPERTIES\s+SOVERSION \$\{JLCS_ABI_VERSION\}\s+VERSION \$\{JLCS_VERSION\}\)", cmake)
    @test at_library !== nothing
    @test at_soname !== nothing
    @test at_library !== nothing && at_soname !== nothing && first(at_library) < first(at_soname)
    # And nothing of ours names LLVM's version in it.
    @test !occursin(r"SOVERSION\s+\$\{LLVM", cmake)
end

@testset "which loaded files are a libJLCS" begin
    yes = ("libJLCS.so", "libJLCS.so.3", "libJLCS.so.3.0.1", "libJLCS.so.23.1",
           "libJLCS.so.23.1jl", "libJLCS.dll", "libJLCS.dylib")
    no = ("libJLCS_thunks.so", "libaot_test_thunks.so", "libJLCS.a", "JLCS.so",
          "libLLVM.so.23.1", "libmyJLCS.so", "libJLCS_thunks.dll")
    for name in yes
        @test _SN_JM._is_jlcs_file(joinpath("somewhere", name))
    end
    for name in no
        @test !_SN_JM._is_jlcs_file(joinpath("somewhere", name))
    end
end

@testset "a second copy is a different FILE, not a different name" begin
    mktempdir() do dir
        real = joinpath(dir, "libJLCS.so.3.0.1")
        write(real, "the library")
        # The library is reached through links; Windows needs a privilege for
        # one, and has no versioned names to link.
        so3 = joinpath(dir, "libJLCS.so.3")
        link = joinpath(dir, "libJLCS.so")
        linked = try
            symlink("libJLCS.so.3.0.1", so3)
            symlink("libJLCS.so.3", link)
            true
        catch
            false
        end
        @test linked || Sys.iswindows()
        own = linked ? link : real

        old = joinpath(mkdir(joinpath(dir, "old")), "libJLCS.so.23.1")
        write(old, "another build")
        thunks = joinpath(dir, "libpkg_thunks.so")
        write(thunks, "a wrapped project's thunks")

        # The same file under each of its names is not a second copy.
        names = linked ? [link, so3, real] : [real]
        @test isempty(_SN_JM._foreign_jlcs(own, names))
        # Another file is, whatever else is loaded.
        loaded = vcat(names, [old, thunks, joinpath(dir, "libc.so.6")])
        @test _SN_JM._foreign_jlcs(own, loaded) == [old]
        # One whose file is gone (deleted while mapped) is compared by path.
        gone = joinpath(dir, "gone", "libJLCS.so.9")
        @test _SN_JM._foreign_jlcs(own, [gone]) == [gone]
        # Nothing loaded, nothing found.
        @test isempty(_SN_JM._foreign_jlcs(own, String[]))
    end
end

@testset "the refusal names both files and what to do" begin
    msg = _SN_JM._second_jlcs_message("/pkg/julia/libpkg_thunks.so", "/new/lib/libJLCS.so",
                                      ["/old/lib/libJLCS.so.23.1jl"])
    @test occursin("/pkg/julia/libpkg_thunks.so", msg)
    @test occursin("/new/lib/libJLCS.so", msg)
    @test occursin("/old/lib/libJLCS.so.23.1jl", msg)
    @test occursin("RepliBuild.wrap", msg)
    @test occursin("exception", msg)
end

@testset "open_thunks_library looks before and after the open" begin
    jm = read(joinpath(RepliBuild.SRC_DIR, "IRGen", "JITManager.jl"), String)
    i = findfirst("function open_thunks_library", jm)
    @test i !== nothing
    body = jm[first(i):end]
    body = body[1:first(something(findfirst("\nend\n", body), (lastindex(body),)))]

    at_before = findfirst("before = _foreign_jlcs(jlcs)", body)
    at_open = findfirst("dlopen(thunks_path", body)
    at_after = findfirst("setdiff(_foreign_jlcs(jlcs), before)", body)
    @test at_before !== nothing && at_open !== nothing && at_after !== nothing
    if at_before !== nothing && at_open !== nothing && at_after !== nothing
        # What was loaded before the open is not this library's doing, and
        # what appeared with it is.
        @test first(at_before) < first(at_open) < first(at_after)
    end
    # A second copy is an error, never a warning the caller reads past.
    @test occursin("error(_second_jlcs_message(thunks_path, jlcs, brought))", body)
end
