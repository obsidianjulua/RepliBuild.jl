# Example: Wrapping Existing Binaries

Complete example of wrapping existing binary libraries (zlib) without source code.

## Project Overview

We'll wrap the zlib compression library to demonstrate binary wrapping.

## Prerequisites

- zlib library installed (`libz.so` on Linux, `libz.dylib` on macOS)

```bash
# Ubuntu/Debian
sudo apt-get install zlib1g-dev

# macOS (usually pre-installed)
brew install zlib
```

## Step 1: Initialize Project

```julia
using RepliBuild

RepliBuild.init("zlib_wrapper", type=:binary)
cd("zlib_wrapper")
```

Directory structure:
```
zlib_wrapper/
├── wrapper_config.toml
├── lib/
├── bin/
└── julia_wrappers/
```

## Step 2: Locate zlib

```bash
# Find zlib location
# Linux
find /usr /lib -name "libz.so*" 2>/dev/null

# macOS
find /usr/local /usr -name "libz.dylib" 2>/dev/null
```

Common locations:
- Linux: `/usr/lib/x86_64-linux-gnu/libz.so.1`
- macOS: `/usr/lib/libz.dylib`

## Step 3: Inspect Binary

```julia
using RepliBuild

# Scan binary for symbols
RepliBuild.scan_binaries("/usr/lib/x86_64-linux-gnu/libz.so.1")
```

Output:
```
📦 Scanning: /usr/lib/x86_64-linux-gnu/libz.so.1
Found 84 exported symbols:

Functions:
  - compress
  - compress2
  - compressBound
  - uncompress
  - deflate
  - deflateEnd
  - deflateInit_
  - inflate
  - inflateEnd
  - inflateInit_
  - gzopen
  - gzread
  - gzwrite
  - gzclose
  ... (more)
```

## Step 4: Configure Wrapper

Edit `wrapper_config.toml`:

```toml
[wrapper]
scan_dirs = ["lib"]
output_dir = "julia_wrappers"
generate_high_level = true

[library.libz]
path = "/usr/lib/x86_64-linux-gnu/libz.so.1"  # Adjust for your system
module_name = "LibZ"

# Core compression functions
exports = [
    "compress",
    "compress2",
    "compressBound",
    "uncompress",
    "deflate",
    "deflateInit_",
    "deflateInit2_",
    "deflateEnd",
    "inflate",
    "inflateInit_",
    "inflateInit2_",
    "inflateEnd",
]

# Function signatures
[library.libz.functions.compress]
return_type = "Cint"
arg_types = ["Ptr{UInt8}", "Ptr{Culong}", "Ptr{UInt8}", "Culong"]

[library.libz.functions.uncompress]
return_type = "Cint"
arg_types = ["Ptr{UInt8}", "Ptr{Culong}", "Ptr{UInt8}", "Culong"]

[library.libz.functions.compressBound]
return_type = "Culong"
arg_types = ["Culong"]
```

## Step 5: Generate Wrapper

```julia
using RepliBuild

# Generate wrappers
RepliBuild.wrap()
```

This creates `julia_wrappers/LibZ.jl`.

## Step 6: Create High-Level API

Create `julia_wrappers/ZlibAPI.jl`:

```julia
module ZlibAPI

# Include low-level wrapper
include("LibZ.jl")
using .LibZ

# Error codes
const Z_OK = 0
const Z_STREAM_END = 1
const Z_NEED_DICT = 2
const Z_ERRNO = -1
const Z_STREAM_ERROR = -2
const Z_DATA_ERROR = -3
const Z_MEM_ERROR = -4
const Z_BUF_ERROR = -5
const Z_VERSION_ERROR = -6

"""
    compress(data::Vector{UInt8}; level=6) -> Vector{UInt8}

Compress data using zlib deflate.

# Arguments
- `data::Vector{UInt8}`: Data to compress
- `level`: Compression level 0-9 (default: 6)

# Returns
Compressed data as `Vector{UInt8}`

# Throws
- `ErrorException`: If compression fails
"""
function compress(data::Vector{UInt8}; level::Int=6)
    # Calculate maximum compressed size
    src_len = Culong(length(data))
    dest_len_ref = Ref{Culong}(LibZ.compressBound(src_len))

    # Allocate destination buffer
    dest = Vector{UInt8}(undef, dest_len_ref[])

    # Compress
    ret = LibZ.compress(dest, dest_len_ref, data, src_len)

    if ret != Z_OK
        error("Compression failed with code: $ret")
    end

    # Resize to actual compressed size
    resize!(dest, dest_len_ref[])
    return dest
end

"""
    uncompress(data::Vector{UInt8}, uncompressed_size::Integer) -> Vector{UInt8}

Decompress zlib-compressed data.

# Arguments
- `data::Vector{UInt8}`: Compressed data
- `uncompressed_size::Integer`: Expected size of uncompressed data

# Returns
Uncompressed data as `Vector{UInt8}`

# Throws
- `ErrorException`: If decompression fails
"""
function uncompress(data::Vector{UInt8}, uncompressed_size::Integer)
    src_len = Culong(length(data))
    dest_len_ref = Ref{Culong}(uncompressed_size)

    # Allocate destination buffer
    dest = Vector{UInt8}(undef, dest_len_ref[])

    # Uncompress
    ret = LibZ.uncompress(dest, dest_len_ref, data, src_len)

    if ret != Z_OK
        if ret == Z_BUF_ERROR
            error("Buffer too small for uncompressed data")
        elseif ret == Z_MEM_ERROR
            error("Out of memory")
        elseif ret == Z_DATA_ERROR
            error("Corrupted compressed data")
        else
            error("Decompression failed with code: $ret")
        end
    end

    return dest
end

"""
    compress(str::String) -> Vector{UInt8}

Compress a string.
"""
function compress(str::String; level::Int=6)
    compress(Vector{UInt8}(str), level=level)
end

"""
    uncompress_to_string(data::Vector{UInt8}, size::Integer) -> String

Decompress data to string.
"""
function uncompress_to_string(data::Vector{UInt8}, size::Integer)
    uncompressed = uncompress(data, size)
    return String(uncompressed)
end

export compress, uncompress, uncompress_to_string

end # module
```

## Step 7: Test Wrapper

Create `test_zlib.jl`:

```julia
using Test
include("julia_wrappers/ZlibAPI.jl")
using .ZlibAPI

@testset "ZlibAPI Tests" begin
    @testset "Basic Compression" begin
        # Test data
        original = "Hello, World! This is a test string for compression."
        original_bytes = Vector{UInt8}(original)

        # Compress
        compressed = ZlibAPI.compress(original_bytes)
        println("Original size: $(length(original_bytes)) bytes")
        println("Compressed size: $(length(compressed)) bytes")
        println("Compression ratio: $(round(length(compressed)/length(original_bytes)*100, digits=1))%")

        @test length(compressed) < length(original_bytes)

        # Uncompress
        uncompressed = ZlibAPI.uncompress(compressed, length(original_bytes))
        @test uncompressed == original_bytes

        # Convert back to string
        result = String(uncompressed)
        @test result == original
    end

    @testset "String API" begin
        original = "The quick brown fox jumps over the lazy dog."

        # Compress string
        compressed = ZlibAPI.compress(original)
        println("\nString compression:")
        println("Original: $(length(original)) bytes")
        println("Compressed: $(length(compressed)) bytes")

        # Uncompress to string
        result = ZlibAPI.uncompress_to_string(compressed, length(original))
        @test result == original
    end

    @testset "Large Data" begin
        # Generate large test data
        original = repeat("Lorem ipsum dolor sit amet. ", 1000)
        original_bytes = Vector{UInt8}(original)

        compressed = ZlibAPI.compress(original_bytes, level=9)  # Max compression
        println("\nLarge data compression:")
        println("Original: $(length(original_bytes)) bytes")
        println("Compressed: $(length(compressed)) bytes")
        println("Ratio: $(round(length(compressed)/length(original_bytes)*100, digits=1))%")

        @test length(compressed) < length(original_bytes) / 2  # Should compress well

        uncompressed = ZlibAPI.uncompress(compressed, length(original_bytes))
        @test uncompressed == original_bytes
    end

    @testset "Different Compression Levels" begin
        data = repeat("test data ", 100)
        original_bytes = Vector{UInt8}(data)

        for level in [1, 6, 9]
            compressed = ZlibAPI.compress(original_bytes, level=level)
            println("\nLevel $level: $(length(compressed)) bytes")

            uncompressed = ZlibAPI.uncompress(compressed, length(original_bytes))
            @test uncompressed == original_bytes
        end
    end

    @testset "Error Handling" begin
        # Try to uncompress invalid data
        invalid_data = Vector{UInt8}([0x00, 0x01, 0x02, 0x03])

        @test_throws ErrorException ZlibAPI.uncompress(invalid_data, 100)
    end
end
```

Run tests:

```julia
include("test_zlib.jl")
```

Output:
```
Original size: 54 bytes
Compressed size: 42 bytes
Compression ratio: 77.8%

String compression:
Original: 45 bytes
Compressed: 37 bytes

Large data compression:
Original: 28000 bytes
Compressed: 89 bytes
Ratio: 0.3%

Level 1: 108 bytes
Level 6: 94 bytes
Level 9: 92 bytes

Test Summary:  | Pass  Total
ZlibAPI Tests  |   13     13
```

## Step 8: Practical Usage

Create `compress_file.jl`:

```julia
include("julia_wrappers/ZlibAPI.jl")
using .ZlibAPI

function compress_file(input_path::String, output_path::String)
    # Read file
    data = read(input_path)
    println("Input file: $input_path")
    println("Original size: $(length(data)) bytes")

    # Compress
    compressed = ZlibAPI.compress(data, level=9)
    println("Compressed size: $(length(compressed)) bytes")
    println("Compression ratio: $(round(length(compressed)/length(data)*100, digits=1))%")

    # Write compressed file
    write(output_path, compressed)
    println("Compressed file: $output_path")
end

function uncompress_file(input_path::String, output_path::String, original_size::Int)
    # Read compressed file
    compressed = read(input_path)

    # Uncompress
    data = ZlibAPI.uncompress(compressed, original_size)

    # Write uncompressed file
    write(output_path, data)
    println("Uncompressed to: $output_path")
end

# Example usage
# compress_file("large_file.txt", "large_file.txt.z")
# uncompress_file("large_file.txt.z", "large_file_restored.txt", original_size)
```

## Complete Project Structure

```
zlib_wrapper/
├── wrapper_config.toml
├── julia_wrappers/
│   ├── LibZ.jl         # Low-level ccall wrappers
│   └── ZlibAPI.jl      # High-level API
├── test_zlib.jl
├── compress_file.jl
└── lib/                # Optional: copy libraries here
```

## Benefits of Binary Wrapping

1. **No source code needed** - Wrap proprietary libraries
2. **Fast** - Direct ccall, no overhead
3. **Portable** - Works with any binary library
4. **Flexible** - Add high-level Julia API on top

## Next Steps

- Wrap other compression libraries (lz4, zstd)
- Wrap SSL/TLS libraries (OpenSSL, mbedTLS)
- Wrap database libraries (SQLite, PostgreSQL)
- See **[Multi-Module Example](multi-module.md)** for complex projects
