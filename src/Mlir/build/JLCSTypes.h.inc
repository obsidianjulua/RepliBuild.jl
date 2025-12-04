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
/// Universal strided array descriptor for cross-language arrays.
/// Canonical representation of multi-dimensional arrays compatible with:
///     - Julia Array{T,N}
///     - NumPy ndarray
///     - C++ strided arrays
///     - Rust slices
/// 
///     Runtime memory layout (as C struct):
///       struct ArrayView {
///         T* data_ptr;        // offset 0:  pointer to element data
///         int64_t* dims_ptr;  // offset 8:  pointer to dimension sizes
///         int64_t* strides_ptr; // offset 16: pointer to stride values (in elements)
///         int64_t rank;       // offset 24: number of dimensions
///       };
/// 
///     This type represents a *view* into array data with explicit strides,
///     enabling zero-copy interop across language boundaries.
class ArrayViewType;
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
namespace detail {
struct ArrayViewTypeStorage;
} // namespace detail
class ArrayViewType : public ::mlir::Type::TypeBase<ArrayViewType, ::mlir::Type, detail::ArrayViewTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "jlcs.array_view";
  static constexpr ::llvm::StringLiteral dialectName = "jlcs";
  static ArrayViewType get(::mlir::MLIRContext *context, Type elementType, unsigned rank);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"array_view"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getElementType() const;
  unsigned getRank() const;
};
} // namespace jlcs
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::jlcs::CStructType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::jlcs::ArrayViewType)

#endif  // GET_TYPEDEF_CLASSES

