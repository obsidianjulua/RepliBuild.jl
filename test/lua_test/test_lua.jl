using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using RepliBuild
using Test

@testset "Lua 5.4 Wrap Test" begin
    println("\n" * "="^70)
    println("Building and Wrapping Lua 5.4...")
    println("="^70)

    toml_path = joinpath(@__DIR__, "replibuild.toml")
    @test isfile(toml_path)

    # Build
    library_path = RepliBuild.build(toml_path)
    @test isfile(library_path)
    println("Library built: $library_path")

    # Wrap
    wrapper_path = RepliBuild.wrap(toml_path)
    @test isfile(wrapper_path)
    println("Wrapper generated: $wrapper_path")

    # Load wrapper
    include(wrapper_path)

    # =========================================================================
    # Test 1: Create a Lua state and check version
    # =========================================================================
    @testset "State creation" begin
        println("\n1. Creating Lua state...")
        L = LuaTest.luaL_newstate()
        @test L != C_NULL
        println("   State created: $L")

        # lua_version returns the Lua version number
        ver = LuaTest.lua_version(L)
        @test ver >= 504.0
        println("   Lua version: $ver")

        LuaTest.lua_close(L)
        println("   State closed.")
    end

    # =========================================================================
    # Test 2: Push and get integers
    # =========================================================================
    @testset "Integer push/get" begin
        println("\n2. Integer stack operations...")
        L = LuaTest.luaL_newstate()

        LuaTest.lua_pushinteger(L, Int64(42))
        @test LuaTest.lua_gettop(L) == 1
        @test LuaTest.lua_isinteger(L, Int32(-1)) == 1

        val = LuaTest.lua_tointegerx(L, Int32(-1), C_NULL)
        @test val == 42
        println("   Pushed 42, got back: $val")

        LuaTest.lua_close(L)
    end

    # =========================================================================
    # Test 3: Push and get strings
    # =========================================================================
    @testset "String push/get" begin
        println("\n3. String stack operations...")
        L = LuaTest.luaL_newstate()

        LuaTest.lua_pushstring(L, "hello from Julia")
        @test LuaTest.lua_isstring(L, Int32(-1)) == 1

        str = LuaTest.lua_tolstring(L, Int32(-1), C_NULL)
        @test str == "hello from Julia"
        println("   Pushed string, got back: \"$str\"")

        LuaTest.lua_close(L)
    end

    # =========================================================================
    # Test 4: Evaluate Lua code (6 * 7 = 42)
    # =========================================================================
    @testset "Eval Lua code" begin
        println("\n4. Evaluating Lua code...")
        L = LuaTest.luaL_newstate()
        LuaTest.luaL_openlibs(L)

        # luaL_loadstring + lua_pcallk = luaL_dostring (which is a macro)
        code = "return 6 * 7"
        load_status = LuaTest.luaL_loadstring(L, code)
        @test load_status == 0  # LUA_OK

        # lua_pcallk(L, nargs=0, nresults=1, errfunc=0, ctx=0, k=NULL)
        call_status = LuaTest.lua_pcallk(L, Int32(0), Int32(1), Int32(0), Int64(0), C_NULL)
        @test call_status == 0  # LUA_OK

        result = LuaTest.lua_tointegerx(L, Int32(-1), C_NULL)
        @test result == 42
        println("   Lua: 'return 6 * 7' = $result")

        LuaTest.lua_close(L)
    end

    # =========================================================================
    # Test 5: Lua string operations
    # =========================================================================
    @testset "Lua string eval" begin
        println("\n5. Lua string operations...")
        L = LuaTest.luaL_newstate()
        LuaTest.luaL_openlibs(L)

        code = """return string.upper("hello lua")"""
        load_status = LuaTest.luaL_loadstring(L, code)
        @test load_status == 0

        call_status = LuaTest.lua_pcallk(L, Int32(0), Int32(1), Int32(0), Int64(0), C_NULL)
        @test call_status == 0

        str = LuaTest.lua_tolstring(L, Int32(-1), C_NULL)
        @test str == "HELLO LUA"
        println("   Lua: string.upper(\"hello lua\") = \"$str\"")

        LuaTest.lua_close(L)
    end

    # =========================================================================
    # Test 6: Table creation and access
    # =========================================================================
    @testset "Table operations" begin
        println("\n6. Table operations...")
        L = LuaTest.luaL_newstate()
        LuaTest.luaL_openlibs(L)

        # Create table via Lua code and read back
        code = """
            t = {x = 10, y = 20}
            return t.x + t.y
        """
        load_status = LuaTest.luaL_loadstring(L, code)
        @test load_status == 0

        call_status = LuaTest.lua_pcallk(L, Int32(0), Int32(1), Int32(0), Int64(0), C_NULL)
        @test call_status == 0

        result = LuaTest.lua_tointegerx(L, Int32(-1), C_NULL)
        @test result == 30
        println("   Lua: t.x + t.y = $result")

        LuaTest.lua_close(L)
    end

    # =========================================================================
    # Test 7: Lua math library
    # =========================================================================
    @testset "Math library" begin
        println("\n7. Lua math library...")
        L = LuaTest.luaL_newstate()
        LuaTest.luaL_openlibs(L)

        code = "return math.sqrt(144)"
        load_status = LuaTest.luaL_loadstring(L, code)
        @test load_status == 0

        call_status = LuaTest.lua_pcallk(L, Int32(0), Int32(1), Int32(0), Int64(0), C_NULL)
        @test call_status == 0

        result = LuaTest.lua_tonumberx(L, Int32(-1), C_NULL)
        @test result == 12.0
        println("   Lua: math.sqrt(144) = $result")

        LuaTest.lua_close(L)
    end

    # =========================================================================
    # Test 8: Julia callback into Lua
    # =========================================================================
    @testset "Julia callback" begin
        println("\n8. Julia -> Lua -> Julia callback...")
        L = LuaTest.luaL_newstate()
        LuaTest.luaL_openlibs(L)

        # Define a Julia function that Lua can call
        # lua_CFunction signature: (lua_State*) -> Cint
        function julia_add(L_ptr::Ptr{Cvoid})::Cint
            a = LuaTest.lua_tointegerx(L_ptr, Int32(1), C_NULL)
            b = LuaTest.lua_tointegerx(L_ptr, Int32(2), C_NULL)
            LuaTest.lua_pushinteger(L_ptr, a + b)
            return Cint(1)  # number of return values
        end

        # Create C function pointer and register as global "julia_add"
        c_func = @cfunction($julia_add, Cint, (Ptr{Cvoid},))
        LuaTest.lua_pushcclosure(L, c_func, Int32(0))
        LuaTest.lua_setglobal(L, "julia_add")

        # Call it from Lua
        code = "return julia_add(100, 200)"
        load_status = LuaTest.luaL_loadstring(L, code)
        @test load_status == 0

        call_status = LuaTest.lua_pcallk(L, Int32(0), Int32(1), Int32(0), Int64(0), C_NULL)
        @test call_status == 0

        result = LuaTest.lua_tointegerx(L, Int32(-1), C_NULL)
        @test result == 300
        println("   Lua called julia_add(100, 200) = $result")

        LuaTest.lua_close(L)
    end

    # =========================================================================
    # Test 9: Multiple return values
    # =========================================================================
    @testset "Multiple returns" begin
        println("\n9. Multiple return values...")
        L = LuaTest.luaL_newstate()
        LuaTest.luaL_openlibs(L)

        code = "return 10, 20, 30"
        load_status = LuaTest.luaL_loadstring(L, code)
        @test load_status == 0

        # nresults = 3
        call_status = LuaTest.lua_pcallk(L, Int32(0), Int32(3), Int32(0), Int64(0), C_NULL)
        @test call_status == 0

        @test LuaTest.lua_gettop(L) == 3
        v1 = LuaTest.lua_tointegerx(L, Int32(1), C_NULL)
        v2 = LuaTest.lua_tointegerx(L, Int32(2), C_NULL)
        v3 = LuaTest.lua_tointegerx(L, Int32(3), C_NULL)
        @test (v1, v2, v3) == (10, 20, 30)
        println("   Lua: return 10, 20, 30 -> ($v1, $v2, $v3)")

        LuaTest.lua_close(L)
    end

    # =========================================================================
    # Test 10: Coroutines
    # =========================================================================
    @testset "Coroutines" begin
        println("\n10. Lua coroutines...")
        L = LuaTest.luaL_newstate()
        LuaTest.luaL_openlibs(L)

        code = """
            co = coroutine.create(function()
                coroutine.yield(1)
                coroutine.yield(2)
                return 3
            end)
            local ok, v1 = coroutine.resume(co)
            local ok, v2 = coroutine.resume(co)
            local ok, v3 = coroutine.resume(co)
            return v1, v2, v3
        """
        load_status = LuaTest.luaL_loadstring(L, code)
        @test load_status == 0

        call_status = LuaTest.lua_pcallk(L, Int32(0), Int32(3), Int32(0), Int64(0), C_NULL)
        @test call_status == 0

        v1 = LuaTest.lua_tointegerx(L, Int32(1), C_NULL)
        v2 = LuaTest.lua_tointegerx(L, Int32(2), C_NULL)
        v3 = LuaTest.lua_tointegerx(L, Int32(3), C_NULL)
        @test (v1, v2, v3) == (1, 2, 3)
        println("   Coroutine yielded: ($v1, $v2, $v3)")

        LuaTest.lua_close(L)
    end

    println("\n" * "="^70)
    println("All Lua 5.4 tests passed!")
    println("Julia -> C (Lua VM) -> Lua execution verified.")
    println("="^70)
end
