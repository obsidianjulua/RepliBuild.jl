using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using RepliBuild
using Test

@testset "Automated Template Instantiation" begin
    toml_path = joinpath(@__DIR__, "replibuild.toml")
    @test isfile(toml_path)

    # Build
    library_path = RepliBuild.build(toml_path, clean=true)
    @test isfile(library_path)

    # Wrap
    wrapper_path = RepliBuild.wrap(toml_path)
    @test isfile(wrapper_path)

    include(wrapper_path)
    
    @test isdefined(Main, :TemplateTest)
    mod = Main.TemplateTest

    methods_found = string.(names(mod, all=true))
    
    println("Exported names:")
    println(filter(m -> contains(m, "MyBox"), methods_found))
    
    @test length(filter(m -> contains(m, "MyBox"), methods_found)) > 0
end
