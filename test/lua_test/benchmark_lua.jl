using BenchmarkTools

include("julia/LuaWrapper.jl")
using .LuaWrapper

# Setup a global state for some tests
const L_global = LuaWrapper.luaL_newstate()
LuaWrapper.luaL_openlibs(L_global)

# --- 1. Pure CCALL (Baseline) ---
# We will define equivalent ccalls directly to measure the theoretical minimum overhead
const LIBLUA = LuaWrapper.LIBRARY_PATH

@inline function pure_ccall_newstate()
    ccall((:luaL_newstate, LIBLUA), Ptr{Cvoid}, ())
end

@inline function pure_ccall_close(L)
    ccall((:lua_close, LIBLUA), Cvoid, (Ptr{Cvoid},), L)
end

@inline function pure_ccall_getglobal(L, name)
    ccall((:lua_getglobal, LIBLUA), Cint, (Ptr{Cvoid}, Cstring), L, name)
end

@inline function pure_ccall_pushinteger(L, n)
    ccall((:lua_pushinteger, LIBLUA), Cvoid, (Ptr{Cvoid}, Clonglong), L, n)
end

@inline function pure_ccall_pop(L, n)
    ccall((:lua_settop, LIBLUA), Cvoid, (Ptr{Cvoid}, Cint), L, -(n)-1)
end

# --- 2. Benchmark Suites ---

println("--- Benchmarking Lua State Creation ---")
print("Pure ccall:    ")
b1 = @btime begin
    L = pure_ccall_newstate()
    pure_ccall_close(L)
end

print("Enhanced Wrap: ")
b2 = @btime begin
    L = LuaWrapper.luaL_newstate()
    LuaWrapper.lua_close(L)
end


println("\n--- Benchmarking String Passing (lua_getglobal) ---")
print("Pure ccall:    ")
b3 = @btime begin
    pure_ccall_getglobal(L_global, "print")
    pure_ccall_pop(L_global, 1)
end

print("Enhanced Wrap: ")
b4 = @btime begin
    LuaWrapper.lua_getglobal(L_global, "print")
    LuaWrapper.lua_settop(L_global, -2)
end


println("\n--- Benchmarking Integer Passing (lua_pushinteger) ---")
print("Pure ccall:    ")
b5 = @btime begin
    pure_ccall_pushinteger(L_global, 42)
    pure_ccall_pop(L_global, 1)
end

print("Enhanced Wrap: ")
b6 = @btime begin
    LuaWrapper.lua_pushinteger(L_global, 42)
    LuaWrapper.lua_settop(L_global, -2)
end

# Cleanup global state
LuaWrapper.lua_close(L_global)
