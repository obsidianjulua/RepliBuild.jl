## Easy Example for Bindings

```julia

using RepliBuild

cd("RepliBuild.jl/test/test_project")

discover()
build("replibuild.toml")
wrap()

```

## Check the Accuracy

"RepliBuild.jl/test/test_project/julia/Project.jl"

Use the bindings in julia by loading the module.

```julia

include("test_project/julia/Project.jl")

```


