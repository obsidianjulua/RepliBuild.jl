#!/usr/bin/env julia
# Wrapper.jl - Enterprise-grade Julia binding generation for compiled libraries
# Three-tier wrapping: Basic (symbol-only) → Advanced (header-aware) → Introspective (metadata-rich)

module Wrapper

using Dates
using JSON
using Libdl   # Tier-1 slice symbol pre-flight (GeneratorC._tier1_preflight!)

# Import from parent RepliBuild module
import ..ConfigurationManager: RepliBuildConfig, get_output_path, get_module_name,
                                get_build_path, get_cache_path
import ..Slicer
import ..ClangJLBridge
import ..BuildBridge
import ..MLIRNative
import ..DWARFParser
import ..JLCSIRGenerator
import ..DAGDiff

export wrap_library, wrap_basic, extract_symbols
export TypeRegistry, SymbolInfo, ParamInfo
export TypeStrictness, STRICT, WARN, PERMISSIVE
export is_struct_like, is_enum_like, is_function_pointer_like

# Sub-modules
include("Wrapper/Utils.jl")
include("Wrapper/C/UtilsC.jl")
include("Wrapper/Cpp/UtilsCpp.jl")
include("Wrapper/TypeRegistry.jl")
include("Wrapper/C/TypesC.jl")
include("Wrapper/Cpp/TypesCpp.jl")
include("Wrapper/DispatchLogic.jl")
include("Wrapper/Symbols.jl")
include("Wrapper/C/IdentifiersC.jl")
include("Wrapper/Cpp/IdentifiersCpp.jl")
include("Wrapper/FunctionPointers.jl")
# Tier 1 (llvmcall over bitcode slices) — experimental, `[wrap.tier1] enable`
# defaults false. Isolated so the whole surface is one file to work on, one
# file to revert, and one guarded call to skip. Must precede GeneratorC.jl,
# which calls into it.
include("Wrapper/C/Tier1C.jl")
include("Wrapper/C/GeneratorC.jl")
include("Wrapper/Cpp/GeneratorCpp.jl")
include("Wrapper/Generator.jl")

end # module Wrapper
