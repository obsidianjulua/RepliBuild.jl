# Builder.jl — Build mechanics: config, environment, compile, link, metadata
# Each file defines its own module; this shim just orchestrates include order.

include("Builder/LLVMEnvironment.jl")
include("Builder/ConfigurationManager.jl")
include("Builder/BuildBridge.jl")
include("Builder/DependencyResolver.jl")
include("Builder/ASTWalker.jl")
include("Builder/Discovery.jl")
include("Builder/ClangJLBridge.jl")
include("Builder/Compiler.jl")
include("Builder/DWARFParser.jl")
include("Builder/EnvironmentDoctor.jl")
include("Builder/PackageRegistry.jl")
