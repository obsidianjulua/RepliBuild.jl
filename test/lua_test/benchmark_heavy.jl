using BenchmarkTools

include("julia/LuaWrapper.jl")
using .LuaWrapper

const L_global = LuaWrapper.luaL_newstate()
LuaWrapper.luaL_openlibs(L_global)

const LIBLUA = LuaWrapper.LIBRARY_PATH

@inline function pure_ccall_loadstring(L, s)
    ccall((:luaL_loadstring, LIBLUA), Cint, (Ptr{Cvoid}, Cstring), L, s)
end

@inline function pure_ccall_pcall(L, nargs, nresults, errfunc)
    ccall((:lua_pcallk, LIBLUA), Cint, (Ptr{Cvoid}, Cint, Cint, Cint, Cint, Ptr{Cvoid}), L, nargs, nresults, errfunc, 0, C_NULL)
end

@inline function pure_ccall_tointeger(L, idx)
    ccall((:lua_tointegerx, LIBLUA), Clonglong, (Ptr{Cvoid}, Cint, Ptr{Cint}), L, idx, C_NULL)
end

@inline function pure_ccall_settop(L, idx)
    ccall((:lua_settop, LIBLUA), Cvoid, (Ptr{Cvoid}, Cint), L, idx)
end

const lua_script = """
function fib(n)
    if n < 2 then return n end
    return fib(n-1) + fib(n-2)
end
return fib(20)
"""

println("--- Benchmarking Heavy Script Evaluation (Fibonacci 20) ---")

print("Pure ccall:    ")
b_heavy_pure = @btime begin
    pure_ccall_loadstring(L_global, lua_script)
    pure_ccall_pcall(L_global, 0, 1, 0)
    res = pure_ccall_tointeger(L_global, -1)
    pure_ccall_settop(L_global, -2)
    res
end

print("Enhanced Wrap: ")
b_heavy_wrap = @btime begin
    LuaWrapper.luaL_loadstring(L_global, lua_script)
    LuaWrapper.lua_pcallk(L_global, 0, 1, 0, LuaWrapper.Cintptr_t(0), C_NULL)
    res = LuaWrapper.lua_tointegerx(L_global, -1, C_NULL)
    LuaWrapper.lua_settop(L_global, -2)
    res
end

# Cleanup global state
LuaWrapper.lua_close(L_global)
