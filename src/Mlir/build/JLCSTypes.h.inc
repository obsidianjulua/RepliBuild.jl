/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace jlcs {
/// A C-ABI-compatible struct with explicit field types and offsets.
class CStructType;
namespace detail {
struct CStructTypeStorage;
} // namespace detail
class CStructType : public ::mlir::Type::TypeBase<CStructType, ::mlir::Type, detail::CStructTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "jlcs.c_struct";
  static constexpr ::llvm::StringLiteral dialectName = "jlcs";
  static CStructType get(::mlir::MLIRContext *context, StringAttr juliaTypeName, ::llvm::ArrayRef<Type> fieldTypes, ArrayAttr fieldOffsets);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"c_struct"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  StringAttr getJuliaTypeName() const;
  ::llvm::ArrayRef<Type> getFieldTypes() const;
  ArrayAttr getFieldOffsets() const;
};
} // namespace jlcs
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::jlcs::CStructType)

#endif  // GET_TYPEDEF_CLASSES

