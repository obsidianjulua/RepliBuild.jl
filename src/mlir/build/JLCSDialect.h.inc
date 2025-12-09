/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: JLCS.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace jlcs {

/// Julia C-Struct layout & FFE dialect
/// Dialect that models C-layouted Julia structs and FFE (Foreign Function Execution)
///     semantics for interop and ABI-aware lowering.
class JLCSDialect : public ::mlir::Dialect {
  explicit JLCSDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~JLCSDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("jlcs");
  }

    void registerTypes();
    void registerOps();
  };
} // namespace jlcs
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::jlcs::JLCSDialect)
