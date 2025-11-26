using RepliBuild

registry = RepliBuild.Wrapper.TypeRegistry(
    Dict("int" => "Cint"),
    Dict(),
    Dict(),
    "Ptr",
    "Ref",
    :strip,
    nothing,
    RepliBuild.Wrapper.WARN,
    true,
    false,
    true
)

empty_params = Vector{RepliBuild.Wrapper.ParamInfo}()

try
    result = RepliBuild.Wrapper.create_symbol_info(
        "_Z10matrix_sum9Matrix3x3",
        :function,
        registry,
        "matrix_sum(Matrix3x3)",
        "void",
        empty_params
    )
    println("SUCCESS: ", result.name, " => ", result.demangled_name)
catch e
    println("FAILED: ", e)
    rethrow()
end
