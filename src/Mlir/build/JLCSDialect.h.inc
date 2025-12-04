/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: JLCSDialect.td                                                       *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace jlcs {

/// Julia C-Struct Layout and FFI Dialect.
/// The Julia C-Struct Dialect (JLCS) is designed to model Julia struct
///     types in a C-ABI-compatible, flattened layout for interoperability
///     with C++ and FFI tools like Julia ccall.
class JLCSDialect : public ::mlir::Dialect {
  explicit JLCSDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~JLCSDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("jlcs");
  }
};
} // namespace jlcs
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::jlcs::JLCSDialect)
