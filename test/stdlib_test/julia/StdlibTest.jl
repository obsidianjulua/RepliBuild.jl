# Auto-generated Julia wrapper for stdlib_test
# Generated: 2026-01-07 04:21:32
# Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
# Library: libstdlib_test.so
# Metadata: compilation_metadata.json
#
# Type Safety: Excellent (~95%) - Types extracted from DWARF debug info
# Ground truth: Types come from compiled binary, not headers
# Manual edits: Minimal to none required

module StdlibTest

const LIBRARY_PATH = "/home/grim/Desktop/Projects/RepliBuild.jl/test/stdlib_test/julia/libstdlib_test.so"

# Verify library exists
if !isfile(LIBRARY_PATH)
    error("Library not found: $LIBRARY_PATH")
end

# =============================================================================
# Compilation Metadata
# =============================================================================

const METADATA = Dict(
    "llvm_version" => "21.1.6",
    "clang_version" => "clang version 21.1.6",
    "optimization" => "0",
    "target_triple" => "x86_64-unknown-linux-gnu",
    "function_count" => 26,
    "generated_at" => "2026-01-07T04:19:04.613"
)

# =============================================================================
# Opaque Struct Declarations
# =============================================================================

mutable struct _IO_FILE end

# =============================================================================
# Struct Definitions (from DWARF debug info)
# =============================================================================

# C++ struct: DateInfo (7 members)
mutable struct DateInfo
    year::Cint
    month::Cint
    day::Cint
    hour::Cint
    minute::Cint
    second::Cint
    nanoseconds::Clong
end

# C++ struct: FileHandle (4 members)
mutable struct FileHandle
    native_handle::Ptr{_IO_FILE}
    mode::NTuple{4, UInt8}
    is_open::Bool
    last_error::Cint
end

# C++ struct: ListNode (3 members)
mutable struct ListNode
    value::Cint
    next::Ptr{ListNode}
    prev::Ptr{ListNode}
end

# C++ struct: StringWrapper (4 members)
mutable struct StringWrapper
    data::Ptr{UInt8}
    length::Csize_t
    capacity::Csize_t
    owns_data::Bool
end

# C++ struct: ratio<1L, 1000000000L> > (1 members)
mutable struct ratio_1L_1000000000L
    __r::Clong
end

# C++ struct: ratio<1L, 1000L> > (1 members)
mutable struct ratio_1L_1000L
    __r::Clong
end

# C++ struct: ratio<1L, 1L> > (1 members)
mutable struct ratio_1L_1L
    __r::Clong
end

# C++ struct: timespec (2 members)
mutable struct timespec
    tv_sec::Clong
    tv_nsec::Clong
end

# C++ struct: tm (11 members)
mutable struct tm
    tm_sec::Cint
    tm_min::Cint
    tm_hour::Cint
    tm_mday::Cint
    tm_mon::Cint
    tm_year::Cint
    tm_wday::Cint
    tm_yday::Cint
    tm_isdst::Cint
    tm_gmtoff::Clong
    tm_zone::Ptr{UInt8}
end

# C++ struct: LinkedList (3 members)
mutable struct LinkedList
    head::Ptr{ListNode}
    tail::Ptr{ListNode}
    size::Csize_t
end

# C++ struct: ratio<1L, 1000000000L> > > (1 members)
mutable struct ratio_1L_1000000000L
    __d::ratio_1L_1000000000L
end


export file_close, file_flush, file_open, file_read, file_seek, file_tell, file_write, list_clear, list_create, list_destroy, list_find, list_pop_back, list_pop_front, list_push_back, list_push_front, string_append, string_compare, string_concat, string_create, string_destroy, string_duplicate, string_get, time_diff_seconds, time_get_current_local, time_get_current_utc, time_sleep_ms, timespec, ratio_1L_1L, ratio_1L_1000L, ListNode, ratio_1L_1000000000L, tm, StringWrapper, FileHandle, LinkedList, DateInfo

"""
    file_close(handle::Ptr{FileHandle}) -> Cvoid

Wrapper for C++ function: `file_close`

# Arguments
- `handle::Ptr{FileHandle}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `file_close`
- Type safety:  From compilation
"""

function file_close(handle::Ptr{FileHandle})::Cvoid
    ccall((:file_close, LIBRARY_PATH), Cvoid, (Ptr{FileHandle},), handle)
end

"""
    file_flush(handle::Ptr{FileHandle}) -> Cint

Wrapper for C++ function: `file_flush`

# Arguments
- `handle::Ptr{FileHandle}`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `file_flush`
- Type safety:  From compilation
"""

function file_flush(handle::Ptr{FileHandle})::Cint
    ccall((:file_flush, LIBRARY_PATH), Cint, (Ptr{FileHandle},), handle)
end

"""
    file_open(path::Ptr{UInt8}, mode::Ptr{UInt8}) -> Ptr{Cvoid}

Wrapper for C++ function: `file_open`

# Arguments
- `path::Ptr{UInt8}`
- `mode::Ptr{UInt8}`

# Returns
- `Ptr{Cvoid}`

# Metadata
- Mangled symbol: `file_open`
- Type safety:  From compilation
"""

function file_open(path::Ptr{UInt8}, mode::Ptr{UInt8})::Ptr{Cvoid}
    ccall((:file_open, LIBRARY_PATH), Ptr{Cvoid}, (Ptr{UInt8}, Ptr{UInt8},), path, mode)
end

"""
    file_read(handle::Ptr{FileHandle}, buffer::Ptr{UInt8}, size::Csize_t) -> Csize_t

Wrapper for C++ function: `file_read`

# Arguments
- `handle::Ptr{FileHandle}`
- `buffer::Ptr{UInt8}`
- `size::Csize_t`

# Returns
- `Csize_t`

# Metadata
- Mangled symbol: `file_read`
- Type safety:  From compilation
"""

function file_read(handle::Ptr{FileHandle}, buffer::Ptr{UInt8}, size::Csize_t)::Csize_t
    ccall((:file_read, LIBRARY_PATH), Csize_t, (Ptr{FileHandle}, Ptr{UInt8}, Csize_t,), handle, buffer, size)
end

"""
    file_seek(handle::Ptr{FileHandle}, offset::Integer, origin::Integer) -> Cint

Wrapper for C++ function: `file_seek`

# Arguments
- `handle::Ptr{FileHandle}`
- `offset::Clong`
- `origin::Cint`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `file_seek`
- Type safety:  From compilation
"""

function file_seek(handle::Ptr{FileHandle}, offset::Integer, origin::Integer)::Cint
    offset_c = Clong(offset)
    origin_c = Cint(origin)  # Auto-converts with overflow check
    return ccall((:file_seek, LIBRARY_PATH), Cint, (Ptr{FileHandle}, Clong, Cint,), handle, offset_c, origin_c)
end

"""
    file_tell(handle::Ptr{FileHandle}) -> Clong

Wrapper for C++ function: `file_tell`

# Arguments
- `handle::Ptr{FileHandle}`

# Returns
- `Clong`

# Metadata
- Mangled symbol: `file_tell`
- Type safety:  From compilation
"""

function file_tell(handle::Ptr{FileHandle})::Clong
    ccall((:file_tell, LIBRARY_PATH), Clong, (Ptr{FileHandle},), handle)
end

"""
    file_write(handle::Ptr{FileHandle}, data::Ptr{UInt8}, size::Csize_t) -> Csize_t

Wrapper for C++ function: `file_write`

# Arguments
- `handle::Ptr{FileHandle}`
- `data::Ptr{UInt8}`
- `size::Csize_t`

# Returns
- `Csize_t`

# Metadata
- Mangled symbol: `file_write`
- Type safety:  From compilation
"""

function file_write(handle::Ptr{FileHandle}, data::Ptr{UInt8}, size::Csize_t)::Csize_t
    ccall((:file_write, LIBRARY_PATH), Csize_t, (Ptr{FileHandle}, Ptr{UInt8}, Csize_t,), handle, data, size)
end

"""
    list_clear(list::Ptr{LinkedList}) -> Cvoid

Wrapper for C++ function: `list_clear`

# Arguments
- `list::Ptr{LinkedList}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `list_clear`
- Type safety:  From compilation
"""

function list_clear(list::Ptr{LinkedList})::Cvoid
    ccall((:list_clear, LIBRARY_PATH), Cvoid, (Ptr{LinkedList},), list)
end

"""
    list_create() -> LinkedList

Wrapper for C++ function: `list_create`

# Arguments


# Returns
- `LinkedList`

# Metadata
- Mangled symbol: `list_create`
- Type safety:  From compilation
"""

function list_create()::LinkedList
    ccall((:list_create, LIBRARY_PATH), LinkedList, (), )
end

"""
    list_destroy(list::Ptr{LinkedList}) -> Cvoid

Wrapper for C++ function: `list_destroy`

# Arguments
- `list::Ptr{LinkedList}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `list_destroy`
- Type safety:  From compilation
"""

function list_destroy(list::Ptr{LinkedList})::Cvoid
    ccall((:list_destroy, LIBRARY_PATH), Cvoid, (Ptr{LinkedList},), list)
end

"""
    list_find(list::Ptr{LinkedList}, value::Integer) -> Ptr{Cvoid}

Wrapper for C++ function: `list_find`

# Arguments
- `list::Ptr{LinkedList}`
- `value::Cint`

# Returns
- `Ptr{Cvoid}`

# Metadata
- Mangled symbol: `list_find`
- Type safety:  From compilation
"""

function list_find(list::Ptr{LinkedList}, value::Integer)::Ptr{Cvoid}
    value_c = Cint(value)  # Auto-converts with overflow check
    return ccall((:list_find, LIBRARY_PATH), Ptr{Cvoid}, (Ptr{LinkedList}, Cint,), list, value_c)
end

"""
    list_pop_back(list::Ptr{LinkedList}) -> Cint

Wrapper for C++ function: `list_pop_back`

# Arguments
- `list::Ptr{LinkedList}`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `list_pop_back`
- Type safety:  From compilation
"""

function list_pop_back(list::Ptr{LinkedList})::Cint
    ccall((:list_pop_back, LIBRARY_PATH), Cint, (Ptr{LinkedList},), list)
end

"""
    list_pop_front(list::Ptr{LinkedList}) -> Cint

Wrapper for C++ function: `list_pop_front`

# Arguments
- `list::Ptr{LinkedList}`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `list_pop_front`
- Type safety:  From compilation
"""

function list_pop_front(list::Ptr{LinkedList})::Cint
    ccall((:list_pop_front, LIBRARY_PATH), Cint, (Ptr{LinkedList},), list)
end

"""
    list_push_back(list::Ptr{LinkedList}, value::Integer) -> Cvoid

Wrapper for C++ function: `list_push_back`

# Arguments
- `list::Ptr{LinkedList}`
- `value::Cint`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `list_push_back`
- Type safety:  From compilation
"""

function list_push_back(list::Ptr{LinkedList}, value::Integer)::Cvoid
    value_c = Cint(value)  # Auto-converts with overflow check
    return ccall((:list_push_back, LIBRARY_PATH), Cvoid, (Ptr{LinkedList}, Cint,), list, value_c)
end

"""
    list_push_front(list::Ptr{LinkedList}, value::Integer) -> Cvoid

Wrapper for C++ function: `list_push_front`

# Arguments
- `list::Ptr{LinkedList}`
- `value::Cint`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `list_push_front`
- Type safety:  From compilation
"""

function list_push_front(list::Ptr{LinkedList}, value::Integer)::Cvoid
    value_c = Cint(value)  # Auto-converts with overflow check
    return ccall((:list_push_front, LIBRARY_PATH), Cvoid, (Ptr{LinkedList}, Cint,), list, value_c)
end

"""
    string_append(dest::Ptr{StringWrapper}, suffix::Ptr{UInt8}) -> Cvoid

Wrapper for C++ function: `string_append`

# Arguments
- `dest::Ptr{StringWrapper}`
- `suffix::Ptr{UInt8}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `string_append`
- Type safety:  From compilation
"""

function string_append(dest::Ptr{StringWrapper}, suffix::Ptr{UInt8})::Cvoid
    ccall((:string_append, LIBRARY_PATH), Cvoid, (Ptr{StringWrapper}, Ptr{UInt8},), dest, suffix)
end

"""
    string_compare(s1::Ptr{StringWrapper}, s2::Ptr{StringWrapper}) -> Cint

Wrapper for C++ function: `string_compare`

# Arguments
- `s1::Ptr{StringWrapper}`
- `s2::Ptr{StringWrapper}`

# Returns
- `Cint`

# Metadata
- Mangled symbol: `string_compare`
- Type safety:  From compilation
"""

function string_compare(s1::Ptr{StringWrapper}, s2::Ptr{StringWrapper})::Cint
    ccall((:string_compare, LIBRARY_PATH), Cint, (Ptr{StringWrapper}, Ptr{StringWrapper},), s1, s2)
end

"""
    string_concat(dest::Ptr{StringWrapper}, src::Ptr{StringWrapper}) -> Cvoid

Wrapper for C++ function: `string_concat`

# Arguments
- `dest::Ptr{StringWrapper}`
- `src::Ptr{StringWrapper}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `string_concat`
- Type safety:  From compilation
"""

function string_concat(dest::Ptr{StringWrapper}, src::Ptr{StringWrapper})::Cvoid
    ccall((:string_concat, LIBRARY_PATH), Cvoid, (Ptr{StringWrapper}, Ptr{StringWrapper},), dest, src)
end

"""
    string_create(initial_str::Ptr{UInt8}) -> StringWrapper

Wrapper for C++ function: `string_create`

# Arguments
- `initial_str::Ptr{UInt8}`

# Returns
- `StringWrapper`

# Metadata
- Mangled symbol: `string_create`
- Type safety:  From compilation
"""

function string_create(initial_str::Ptr{UInt8})::StringWrapper
    ccall((:string_create, LIBRARY_PATH), StringWrapper, (Ptr{UInt8},), initial_str)
end

"""
    string_destroy(s::Ptr{StringWrapper}) -> Cvoid

Wrapper for C++ function: `string_destroy`

# Arguments
- `s::Ptr{StringWrapper}`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `string_destroy`
- Type safety:  From compilation
"""

function string_destroy(s::Ptr{StringWrapper})::Cvoid
    ccall((:string_destroy, LIBRARY_PATH), Cvoid, (Ptr{StringWrapper},), s)
end

"""
    string_duplicate(src::Ptr{StringWrapper}) -> StringWrapper

Wrapper for C++ function: `string_duplicate`

# Arguments
- `src::Ptr{StringWrapper}`

# Returns
- `StringWrapper`

# Metadata
- Mangled symbol: `string_duplicate`
- Type safety:  From compilation
"""

function string_duplicate(src::Ptr{StringWrapper})::StringWrapper
    ccall((:string_duplicate, LIBRARY_PATH), StringWrapper, (Ptr{StringWrapper},), src)
end

"""
    string_get(s::Ptr{StringWrapper}) -> Cstring

Wrapper for C++ function: `string_get`

# Arguments
- `s::Ptr{StringWrapper}`

# Returns
- `Cstring`

# Metadata
- Mangled symbol: `string_get`
- Type safety:  From compilation
"""

function string_get(s::Ptr{StringWrapper})::String
    ptr = ccall((:string_get, LIBRARY_PATH), Cstring, (Ptr{StringWrapper},), s)
    if ptr == C_NULL
        error("string_get returned NULL pointer")
    end
    return unsafe_string(ptr)
end

"""
    time_diff_seconds(start::DateInfo, end_::DateInfo) -> Cdouble

Wrapper for C++ function: `time_diff_seconds`

# Arguments
- `start::DateInfo`
- `end::DateInfo`

# Returns
- `Cdouble`

# Metadata
- Mangled symbol: `time_diff_seconds`
- Type safety:  From compilation
"""

function time_diff_seconds(start::DateInfo, end_::DateInfo)::Cdouble
    ccall((:time_diff_seconds, LIBRARY_PATH), Cdouble, (DateInfo, DateInfo,), start, end_)
end

"""
    time_get_current_local() -> DateInfo

Wrapper for C++ function: `time_get_current_local`

# Arguments


# Returns
- `DateInfo`

# Metadata
- Mangled symbol: `time_get_current_local`
- Type safety:  From compilation
"""

function time_get_current_local()::DateInfo
    ccall((:time_get_current_local, LIBRARY_PATH), DateInfo, (), )
end

"""
    time_get_current_utc() -> DateInfo

Wrapper for C++ function: `time_get_current_utc`

# Arguments


# Returns
- `DateInfo`

# Metadata
- Mangled symbol: `time_get_current_utc`
- Type safety:  From compilation
"""

function time_get_current_utc()::DateInfo
    ccall((:time_get_current_utc, LIBRARY_PATH), DateInfo, (), )
end

"""
    time_sleep_ms(milliseconds::Cuint) -> Cvoid

Wrapper for C++ function: `time_sleep_ms`

# Arguments
- `milliseconds::Cuint`

# Returns
- `Cvoid`

# Metadata
- Mangled symbol: `time_sleep_ms`
- Type safety:  From compilation
"""

function time_sleep_ms(milliseconds::Cuint)::Cvoid
    ccall((:time_sleep_ms, LIBRARY_PATH), Cvoid, (Cuint,), milliseconds)
end


end # module StdlibTest
