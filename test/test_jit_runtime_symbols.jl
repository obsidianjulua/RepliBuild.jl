#!/usr/bin/env julia
# test/test_jit_runtime_symbols.jl — a thunk's callee that lives in the C++
# runtime library binds through the wrapped library's own handle (2026-11-13).
#
# The JIT searched only the wrapped library, so a member of a listed STL
# container the library calls in libstdc++ rather than instantiating itself
# (`basic_string<char>` is an extern template there) had no thunk. On glibc,
# `dlsym` through a library's OWN handle searches its dependency tree, so the
# member answers through the library that links it. Three pure pieces, so no
# library is needed:
#
#   * `Compiler._nm_symbol_lines`: the undefined (`U`) lines of `nm -g`, their
#     `@GLIBCXX_…` version suffix cut (with it on, dlsym resolves nothing).
#   * `Compiler._stl_methods_from_nm` with `resolvable`: an undefined member
#     the handle resolves enters the table marked `runtime_library`; one it
#     does not is unbindable; the address-keyed listings older tests pass
#     see no undefined symbol.
#   * `JITManager._preflight_callees`: every callee the module will name,
#     through any `lookup`; the library's own are ORC's, a resolved foreign one
#     is bound at that address, an unresolved one is dropped before the module
#     exists, so ORC never meets an unresolved symbol (it blocks, it does not
#     raise). `_check_module_declarations` names any declaration that slipped
#     past.
#
# Negative check: make `_preflight_callees` return every foreign callee as
# unresolved (`lookup = _ -> nothing`) and the "resolved callees are bound"
# testset fails; make `_stl_methods_from_nm` ignore `resolvable` and the
# runtime_library testset fails. The live half is test_stl_runtime_live.jl.

using Test
using RepliBuild

const _RS_C = RepliBuild.Compiler
const _RS_J = RepliBuild.JITManager

const _RS_SIZE = "_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv"
const _RS_CSTR = "_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv"
const _RS_PUSH = "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc"
const _RS_OWN = "_Z5greetPKc"

@testset "runtime-library callees bind through the library's handle" begin
    @testset "nm's undefined lines, version cut" begin
        m = """
        0000000000001160 T _Z5greetPKc
                         U $(_RS_SIZE)@GLIBCXX_3.4.21
                         U $(_RS_CSTR)@GLIBCXX_3.4.21
                         w __gmon_start__
        0000000000004020 D _ZTVN8tinyxml27XMLNodeE
                         U $(_RS_PUSH)@GLIBCXX_3.4.21
        """
        d = """
        0000000000001160 T greet(char const*)
                         U std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::size() const@GLIBCXX_3.4.21
                         U std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::c_str() const@GLIBCXX_3.4.21
                         w __gmon_start__
        0000000000004020 D vtable for tinyxml2::XMLNode
                         U std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::push_back(char)@GLIBCXX_3.4.21
        """
        ml = _RS_C._nm_symbol_lines(m)
        dl = _RS_C._nm_symbol_lines(d)
        @test length(ml) == 4 && length(dl) == 4                   # the weak-undefined and data lines are out
        @test [x[1] for x in ml] == ["T", "U", "U", "U"] == [x[1] for x in dl]
        @test ml[2] == ("U", "", _RS_SIZE)                          # no address, no version
        @test dl[2][3] == "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::size() const"
        @test ml[1] == ("T", "0000000000001160", _RS_OWN)
        @test _RS_C._nm_undefined_names(m) == [_RS_SIZE, _RS_CSTR, _RS_PUSH]

        @testset "runtime members enter the table only when the handle resolves them" begin
            tmpl = ["std::string"]
            (t1, u1) = _RS_C._stl_methods_from_nm(m, d, tmpl; runtime = Set([_RS_OWN]),
                                                 resolvable = Set([_RS_SIZE, _RS_CSTR]))
            ms = get(t1, "std::basic_string<char>", Dict{String,Any}[])
            @test Set(x["method"] for x in ms) == Set(["size", "c_str"])
            @test all(x -> get(x, "runtime_library", false) == true, ms)
            @test Set(x["mangled"] for x in ms) == Set([_RS_SIZE, _RS_CSTR])
            @test u1 == ["std::basic_string<char>::push_back"]       # referenced, not resolvable
            # No probe answer: an undefined symbol is never a candidate.
            (t2, u2) = _RS_C._stl_methods_from_nm(m, d, tmpl; runtime = Set([_RS_OWN]))
            @test isempty(get(t2, "std::basic_string<char>", Dict{String,Any}[])) && isempty(u2)
            # Listings that do not line up (the address-keyed shape): no undefined symbol is reached.
            (t3, _) = _RS_C._stl_methods_from_nm(m, "0000000000001160 T greet(char const*)\n", tmpl;
                                                resolvable = Set([_RS_SIZE]))
            @test isempty(get(t3, "std::basic_string<char>", Dict{String,Any}[]))
        end
    end

    @testset "the pre-flight: own callees are ORC's, resolved foreign ones bound, the rest dropped" begin
        table = Dict(_RS_SIZE => Ptr{Cvoid}(UInt(0x1000)), _RS_CSTR => Ptr{Cvoid}(UInt(0x2000)), _RS_OWN => Ptr{Cvoid}(UInt(0x3000)))
        asked = String[]
        lookup = n -> (push!(asked, n); get(table, n, nothing))
        (bound, unresolved) = _RS_J._preflight_callees(lookup, Set([_RS_OWN, _RS_SIZE, _RS_CSTR, _RS_PUSH]), Set([_RS_OWN]))
        @test bound == Dict(_RS_SIZE => Ptr{Cvoid}(UInt(0x1000)), _RS_CSTR => Ptr{Cvoid}(UInt(0x2000)))
        @test unresolved == [_RS_PUSH]
        @test !(_RS_OWN in asked)                                   # never looked up: the library's own
        @test sort(asked) == sort([_RS_SIZE, _RS_CSTR, _RS_PUSH])
        # A null answer is unresolved too.
        (b2, u2) = _RS_J._preflight_callees(_ -> C_NULL, Set([_RS_SIZE]), Set{String}())
        @test isempty(b2) && u2 == [_RS_SIZE]
    end

    @testset "what the module names, and dropping what nothing resolves" begin
        meta = Dict{String,Any}(
            "functions" => Any[Dict{String,Any}("mangled" => _RS_OWN), Dict{String,Any}("mangled" => "_Z3fooi")],
            "stl_methods" => Dict{String,Any}("std::basic_string<char>" => Any[
                Dict{String,Any}("method" => "size", "mangled" => _RS_SIZE, "runtime_library" => true),
                Dict{String,Any}("method" => "push_back", "mangled" => _RS_PUSH, "runtime_library" => true)]))
        @test _RS_J._module_callees(meta, nothing) == Set([_RS_OWN, "_Z3fooi", _RS_SIZE, _RS_PUSH])
        @test _RS_J._module_callees(meta, Set([_RS_OWN])) == Set([_RS_OWN, _RS_SIZE, _RS_PUSH])
        _RS_J._drop_stl_methods!(meta, [_RS_PUSH])
        @test [x["method"] for x in meta["stl_methods"]["std::basic_string<char>"]] == ["size"]

        ir = """
        func.func private @$(_RS_OWN)(!llvm.ptr) -> i32
        func.func private @$(_RS_SIZE)(!llvm.ptr) -> i64
        func.func private @dispatch__ZN1A1fEv(!llvm.ptr)
        func.func private @jlcs_catch_current_exception() -> !llvm.ptr
        func.func private @_Z7strangev() -> i32
        func.func @$(_RS_OWN)_thunk(%a: !llvm.ptr) -> i32 attributes { llvm.emit_c_interface } { }
        """
        stray = @test_logs (:warn, r"nothing pre-flighted") _RS_J._check_module_declarations(
            ir, Set([_RS_OWN]), Dict(_RS_SIZE => Ptr{Cvoid}(UInt(1))), "libx.so")
        @test stray == ["_Z7strangev"]
        @test isempty(@test_logs _RS_J._check_module_declarations(
            ir, Set([_RS_OWN, "_Z7strangev"]), Dict(_RS_SIZE => Ptr{Cvoid}(UInt(1))), "libx.so"))
    end
end
