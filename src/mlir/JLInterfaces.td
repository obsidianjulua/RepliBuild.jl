// JLInheritanceInterface.td

def JL_SubtypeInterface : TypeInterface<"JuliaSubtype">
{
    let description = "Interface for types that are part of the Julia type hierarchy.";
  let methods = [
    // This method allows us to query the immediate supertype's name.
    InterfaceMethod<
      "GetSupertypeName",
      "::mlir::StringAttr",
      "getSupertypeName",
      (ins "::mlir::MLIRContext *":$ctx),
      [{ return getImpl()->getSupertypeName(cast<ConcreteType>(*this)); }],
      [{
        //! Returns the name of the immediate supertype/base class.
        return ::mlir::StringAttr::get(ctx, "::jl::BaseClass");
      }]
    >
  ];
}

/*
    // Pseudocode C++ Binding Generator Pass

for (auto typeInfoOp : module.getOps<jl::TypeInfoOp>())
{
    mlir::Type structType = typeInfoOp.getStructType();

    // Check if the type implements the interface, NOT if it's a specific type
    if (auto subtypeInterface = structType.dyn_cast<jl::JL_SubtypeInterface>()) {

        // Use the generic interface method to get the base class name
        StringAttr baseClassName = subtypeInterface.GetSupertypeName(structType.getContext());

        // Generate the C++ class with the correct inheritance
        class MyPoint : public baseClassName.getValue() { ... };
    }
}
*/
