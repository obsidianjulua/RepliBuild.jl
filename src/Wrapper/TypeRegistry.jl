# TypeRegistry.jl

"""
    TypeStrictness
"""
@enum TypeStrictness begin
    STRICT = 1
    WARN = 2
    PERMISSIVE = 3
end

"""
    TypeRegistry
"""
struct TypeRegistry
    language::Symbol                     # :c or :cpp
    base_types::Dict{String,String}      
    stl_types::Dict{String,String}       
    custom_types::Dict{String,String}    
    pointer_suffix::String               
    reference_suffix::String             
    const_handling::Symbol               
    compilation_metadata::Union{Nothing,Dict}
    strictness::TypeStrictness           
    allow_unknown_structs::Bool          
    allow_unknown_enums::Bool            
    allow_function_pointers::Bool        
end

function create_type_registry(config; custom_types::Dict{String,String}=Dict{String,String}(),
                              strictness::Union{TypeStrictness,Nothing}=nothing,
                              allow_unknown_structs::Union{Bool,Nothing}=nothing,
                              allow_unknown_enums::Union{Bool,Nothing}=nothing,
                              allow_function_pointers::Union{Bool,Nothing}=nothing)
    types_cfg = config.types

    # Convert Symbol to TypeStrictness
    final_strictness = if !isnothing(strictness)
        strictness
    elseif types_cfg.strictness == :strict
        STRICT
    elseif types_cfg.strictness == :permissive
        PERMISSIVE
    else
        WARN
    end

    final_allow_structs = isnothing(allow_unknown_structs) ? types_cfg.allow_unknown_structs : allow_unknown_structs
    final_allow_enums = isnothing(allow_unknown_enums) ? types_cfg.allow_unknown_enums : allow_unknown_enums
    final_allow_fptrs = isnothing(allow_function_pointers) ? types_cfg.allow_function_pointers : allow_function_pointers
    merged_custom = merge(types_cfg.custom_mappings, custom_types)

    base_types = Dict{String,String}(
        "void" => "Cvoid", "bool" => "Bool", "_Bool" => "Bool",
        "char" => "Cchar", "signed char" => "Cchar", "unsigned char" => "Cuchar",
        "wchar_t" => "Cwchar_t", "char16_t" => "UInt16", "char32_t" => "UInt32",
        "short" => "Cshort", "short int" => "Cshort", "signed short" => "Cshort", "unsigned short" => "Cushort",
        "int" => "Cint", "signed int" => "Cint", "unsigned int" => "Cuint", "unsigned" => "Cuint",
        "long" => "Clong", "long int" => "Clong", "signed long" => "Clong", "unsigned long" => "Culong",
        "long long" => "Clonglong", "long long int" => "Clonglong", "signed long long" => "Clonglong", "unsigned long long" => "Culonglong",
        "int8_t" => "Int8", "uint8_t" => "UInt8", "int16_t" => "Int16", "uint16_t" => "UInt16",
        "int32_t" => "Int32", "uint32_t" => "UInt32", "int64_t" => "Int64", "uint64_t" => "UInt64",
        "float" => "Cfloat", "double" => "Cdouble", "long double" => "Float64",
        "size_t" => "Csize_t", "ssize_t" => "Cssize_t", "ptrdiff_t" => "Cptrdiff_t",
        "intptr_t" => "Int64", "uintptr_t" => "UInt64", "off_t" => "Int64", "time_t" => "Int64", "clock_t" => "Int64",
        # C11 _Complex types
        "_Complex float" => "ComplexF32", "complex float" => "ComplexF32", "float _Complex" => "ComplexF32",
        "_Complex double" => "ComplexF64", "complex double" => "ComplexF64", "double _Complex" => "ComplexF64",
        "_Complex long double" => "ComplexF64", "complex long double" => "ComplexF64", "long double _Complex" => "ComplexF64"
    )

    stl_types = Dict{String,String}(
        "std::string" => "Ptr{Cvoid}", "std::basic_string<char>" => "Ptr{Cvoid}", "std::string_view" => "Ptr{Cvoid}", "std::basic_string_view<char>" => "Ptr{Cvoid}",
        "std::vector" => "Ptr{Cvoid}", "std::array" => "Ptr{Cvoid}", "std::deque" => "Ptr{Cvoid}", "std::list" => "Ptr{Cvoid}", "std::forward_list" => "Ptr{Cvoid}",
        "std::map" => "Ptr{Cvoid}", "std::unordered_map" => "Ptr{Cvoid}", "std::multimap" => "Ptr{Cvoid}", "std::set" => "Ptr{Cvoid}", "std::unordered_set" => "Ptr{Cvoid}", "std::multiset" => "Ptr{Cvoid}",
        "std::pair" => "Tuple", "std::tuple" => "Tuple", "std::optional" => "Union{Nothing,T} where T", "std::unique_ptr" => "Ptr", "std::shared_ptr" => "Ptr", "std::weak_ptr" => "Ptr"
    )

    lang = config.wrap.language

    return TypeRegistry(
        lang,                       # language
        base_types,
        stl_types,
        merged_custom,              # custom_types
        "Ptr",                      # pointer_suffix
        "Ref",                      # reference_suffix
        :strip,                     # const_handling
        nothing,                    # compilation_metadata
        final_strictness,           # strictness
        final_allow_structs,        # allow_unknown_structs
        final_allow_enums,          # allow_unknown_enums
        final_allow_fptrs           # allow_function_pointers
    )
end

function infer_julia_type(registry::TypeRegistry, type_str::String; context::String="")::String
    if registry.language == :c
        return infer_c_type(registry, type_str; context=context)
    else
        return infer_cpp_type(registry, type_str; context=context)
    end
end
