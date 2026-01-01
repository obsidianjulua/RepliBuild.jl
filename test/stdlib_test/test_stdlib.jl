using Pkg
Pkg.activate(".") # Activate the project at root

using RepliBuild
using Test

# Set working directory to test/stdlib_test
cd(@__DIR__)

@testset "StdLib Test" begin
    # 1. Run the RepliBuild pipeline
    @info "Running RepliBuild pipeline..."
    
    # Clean previous run
    rm("build", force=true, recursive=true)
    rm("julia", force=true, recursive=true)
    rm(".replibuild_cache", force=true, recursive=true)

    # Execute pipeline
    RepliBuild.discover()
    RepliBuild.build()
    RepliBuild.wrap()

    # 2. Verify output exists
    @test isfile("julia/StdLibTest.jl")
    @test isfile("julia/libStdLibTest.so")

    # 3. Load the generated module
    include("julia/StdLibTest.jl")
    using .StdLibTest

    @testset "String Operations" begin
        # Create string
        s = string_create("Hello")
        @test string_get(Ref(s)) |> unsafe_string == "Hello"
        @test s.length == 5

        # Append
        string_append(Ref(s), " World")
        @test string_get(Ref(s)) |> unsafe_string == "Hello World"
        @test s.length == 11

        # Duplicate
        s2 = string_duplicate(Ref(s))
        @test string_compare(Ref(s), Ref(s2)) == 0
        
        # Modify copy
        string_append(Ref(s2), "!")
        @test string_compare(Ref(s), Ref(s2)) < 0 # s < s2

        string_destroy(Ref(s))
        string_destroy(Ref(s2))
    end

    @testset "File I/O" begin
        filename = "test_file.txt"
        content = "Hello File I/O"
        
        # Write
        fh = file_open(filename, "w")
        @test fh != C_NULL
        written = file_write(fh, content, length(content))
        @test written == length(content)
        file_close(fh)

        # Read
        fh = file_open(filename, "r")
        @test fh != C_NULL
        buffer = Vector{UInt8}(undef, 100)
        read_len = file_read(fh, pointer(buffer) |> Ptr{Cchar}, 100)
        @test read_len == length(content)
        read_str = unsafe_string(pointer(buffer), read_len)
        @test read_str == content
        file_close(fh)

        rm(filename)
    end

    @testset "Time Operations" begin
        t1 = time_get_current_utc()
        sleep(0.1) # Wait a bit
        t2 = time_get_current_utc()
        
        diff = time_diff_seconds(t1, t2)
        @test diff >= 0.1
        @test diff < 0.2 # Allow some overhead

        # Test local time exists (basic check)
        local_t = time_get_current_local()
        @test local_t.year >= 2025
    end

    @testset "Linked List" begin
        list = list_create()
        @test list.size == 0
        @test list.head == C_NULL
        
        list_push_back(Ref(list), 10)
        list_push_back(Ref(list), 20)
        list_push_front(Ref(list), 5)
        
        @test list.size == 3
        
        # Find
        node = list_find(Ref(list), 20)
        @test node != C_NULL
        # unsafe_load to check value if needed, but simple pointer check is good for now
        
        # Pop
        val = list_pop_back(Ref(list))
        @test val == 20
        @test list.size == 2
        
        val = list_pop_front(Ref(list))
        @test val == 5
        @test list.size == 1
        
        list_destroy(Ref(list))
    end
end
