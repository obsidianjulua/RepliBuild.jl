#!/usr/bin/env julia
# Wrapper.jl - Enterprise-grade Julia binding generation for compiled libraries
# Three-tier wrapping: Basic (symbol-only) → Advanced (header-aware) → Introspective (metadata-rich)

module Wrapper

using Dates
using JSON

# Import from parent RepliBuild module
import ..ConfigurationManager: RepliBuildConfig, get_output_path, get_module_name
import ..ClangJLBridge
import ..BuildBridge
import ..MLIRNative
import ..DWARFParser
import ..JLCSIRGenerator

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
include("Wrapper/C/GeneratorC.jl")
include("Wrapper/Cpp/GeneratorCpp.jl")
include("Wrapper/Generator.jl")

end # module Wrapper
