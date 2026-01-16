/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Interface for types that are part of the Julia type hierarchy.
class JuliaSubtype;
namespace detail {
struct JuliaSubtypeInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    StringAttr (*getSupertypeName)(const Concept *impl, ::mlir::Type , MLIRContext *);
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = JuliaSubtype;
    Model() : Concept{getSupertypeName} {}

    static inline StringAttr getSupertypeName(const Concept *impl, ::mlir::Type tablegen_opaque_val, MLIRContext * ctx);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = JuliaSubtype;
    FallbackModel() : Concept{getSupertypeName} {}

    static inline StringAttr getSupertypeName(const Concept *impl, ::mlir::Type tablegen_opaque_val, MLIRContext * ctx);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
  };
};
template <typename ConcreteType>
struct JuliaSubtypeTrait;

} // namespace detail
class JuliaSubtype : public ::mlir::TypeInterface<JuliaSubtype, detail::JuliaSubtypeInterfaceTraits> {
public:
  using ::mlir::TypeInterface<JuliaSubtype, detail::JuliaSubtypeInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::JuliaSubtypeTrait<ConcreteType> {};

  /// Get the supertype name
  StringAttr getSupertypeName(MLIRContext * ctx) const;
};
namespace detail {
  template <typename ConcreteType>
  struct JuliaSubtypeTrait : public ::mlir::TypeInterface<JuliaSubtype, detail::JuliaSubtypeInterfaceTraits>::Trait<ConcreteType> {
  };
}// namespace detail
template<typename ConcreteType>
StringAttr detail::JuliaSubtypeInterfaceTraits::Model<ConcreteType>::getSupertypeName(const Concept *impl, ::mlir::Type tablegen_opaque_val, MLIRContext * ctx) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getSupertypeName(ctx);
}
template<typename ConcreteType>
StringAttr detail::JuliaSubtypeInterfaceTraits::FallbackModel<ConcreteType>::getSupertypeName(const Concept *impl, ::mlir::Type tablegen_opaque_val, MLIRContext * ctx) {
  return static_cast<const ConcreteType *>(impl)->getSupertypeName(tablegen_opaque_val, ctx);
}
