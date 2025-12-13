/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: JLCS.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace jlcs {
/// Generic operation to read a field from a C-compatible struct.
class GetFieldOp;
} // namespace jlcs
} // namespace mlir
namespace mlir {
namespace jlcs {
/// Call an external C function using FFE metadata
class FFECallOp;
} // namespace jlcs
} // namespace mlir
namespace mlir {
namespace jlcs {
/// Load element from strided array view.
class LoadArrayElementOp;
} // namespace jlcs
} // namespace mlir
namespace mlir {
namespace jlcs {
/// Generic operation to write a field into a C-compatible struct.
class SetFieldOp;
} // namespace jlcs
} // namespace mlir
namespace mlir {
namespace jlcs {
/// Store element to strided array view.
class StoreArrayElementOp;
} // namespace jlcs
} // namespace mlir
namespace mlir {
namespace jlcs {
/// Declares a CStruct type and its C++ base class mapping.
class TypeInfoOp;
} // namespace jlcs
} // namespace mlir
namespace mlir {
namespace jlcs {
/// Call a C++ virtual method through vtable dispatch.
class VirtualCallOp;
} // namespace jlcs
} // namespace mlir
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::GetFieldOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GetFieldOpGenericAdaptorBase {
public:
  struct Properties {
    using fieldOffsetTy = ::mlir::IntegerAttr;
    fieldOffsetTy fieldOffset;

    auto getFieldOffset() const {
      auto &propStorage = this->fieldOffset;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setFieldOffset(const ::mlir::IntegerAttr &propValue) {
      this->fieldOffset = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.fieldOffset == this->fieldOffset &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  GetFieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("jlcs.get_field", odsAttrs.getContext());
  }

  GetFieldOpGenericAdaptorBase(GetFieldOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getFieldOffsetAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().fieldOffset);
    return attr;
  }

  uint64_t getFieldOffset();
};
} // namespace detail
template <typename RangeT>
class GetFieldOpGenericAdaptor : public detail::GetFieldOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GetFieldOpGenericAdaptorBase;
  static constexpr int odsIndex_structValue = 0;
public:
  GetFieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  GetFieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : GetFieldOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  GetFieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : GetFieldOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  GetFieldOpGenericAdaptor(RangeT values, const GetFieldOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = GetFieldOp, typename = std::enable_if_t<std::is_same_v<LateInst, GetFieldOp>>>
  GetFieldOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getStructValue() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GetFieldOpAdaptor : public GetFieldOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GetFieldOpGenericAdaptor::GetFieldOpGenericAdaptor;
  GetFieldOpAdaptor(GetFieldOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class GetFieldOp : public ::mlir::Op<GetFieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GetFieldOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GetFieldOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_structValue = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("fieldOffset")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getFieldOffsetAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getFieldOffsetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("jlcs.get_field");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getStructValue() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getStructValueMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getFieldOffsetAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().fieldOffset);
  }

  uint64_t getFieldOffset();
  void setFieldOffsetAttr(::mlir::IntegerAttr attr) {
    getProperties().fieldOffset = attr;
  }

  void setFieldOffset(uint64_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value structValue, IntegerAttr fieldOffset, Type resultType);
  static GetFieldOp create(::mlir::OpBuilder &builder, ::mlir::Location location, Value structValue, IntegerAttr fieldOffset, Type resultType);
  static GetFieldOp create(::mlir::ImplicitLocOpBuilder &builder, Value structValue, IntegerAttr fieldOffset, Type resultType);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace jlcs
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::jlcs::GetFieldOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::FFECallOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class FFECallOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  FFECallOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("jlcs.ffe_call", odsAttrs.getContext());
  }

  FFECallOpGenericAdaptorBase(FFECallOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};
} // namespace detail
template <typename RangeT>
class FFECallOpGenericAdaptor : public detail::FFECallOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::FFECallOpGenericAdaptorBase;
  static constexpr int odsIndex_args = 0;
public:
  FFECallOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  FFECallOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : FFECallOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  FFECallOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : FFECallOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  FFECallOpGenericAdaptor(RangeT values, const FFECallOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = FFECallOp, typename = std::enable_if_t<std::is_same_v<LateInst, FFECallOp>>>
  FFECallOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getArgs() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class FFECallOpAdaptor : public FFECallOpGenericAdaptor<::mlir::ValueRange> {
public:
  using FFECallOpGenericAdaptor::FFECallOpGenericAdaptor;
  FFECallOpAdaptor(FFECallOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class FFECallOp : public ::mlir::Op<FFECallOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FFECallOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = FFECallOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_args = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("jlcs.ffe_call");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getArgs() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::result_range getResults() {
    return getODSResults(0);
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static FFECallOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static FFECallOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static FFECallOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static FFECallOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace jlcs
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::jlcs::FFECallOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::LoadArrayElementOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class LoadArrayElementOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoadArrayElementOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("jlcs.load_array_element", odsAttrs.getContext());
  }

  LoadArrayElementOpGenericAdaptorBase(LoadArrayElementOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};
} // namespace detail
template <typename RangeT>
class LoadArrayElementOpGenericAdaptor : public detail::LoadArrayElementOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoadArrayElementOpGenericAdaptorBase;
  static constexpr int odsIndex_view = 0;
  static constexpr int odsIndex_indices = 1;
public:
  LoadArrayElementOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoadArrayElementOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoadArrayElementOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoadArrayElementOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoadArrayElementOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoadArrayElementOpGenericAdaptor(RangeT values, const LoadArrayElementOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoadArrayElementOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoadArrayElementOp>>>
  LoadArrayElementOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getView() {
    return (*getODSOperands(0).begin());
  }

  RangeT getIndices() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoadArrayElementOpAdaptor : public LoadArrayElementOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoadArrayElementOpGenericAdaptor::LoadArrayElementOpGenericAdaptor;
  LoadArrayElementOpAdaptor(LoadArrayElementOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoadArrayElementOp : public ::mlir::Op<LoadArrayElementOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoadArrayElementOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoadArrayElementOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_view = 0;
  static constexpr int odsIndex_indices = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("jlcs.load_array_element");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getView() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::Operation::operand_range getIndices() {
    return getODSOperands(1);
  }

  ::mlir::OpOperand &getViewMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value view, ValueRange indices, Type resultType);
  static LoadArrayElementOp create(::mlir::OpBuilder &builder, ::mlir::Location location, Value view, ValueRange indices, Type resultType);
  static LoadArrayElementOp create(::mlir::ImplicitLocOpBuilder &builder, Value view, ValueRange indices, Type resultType);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};
} // namespace jlcs
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::jlcs::LoadArrayElementOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::SetFieldOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SetFieldOpGenericAdaptorBase {
public:
  struct Properties {
    using fieldOffsetTy = ::mlir::IntegerAttr;
    fieldOffsetTy fieldOffset;

    auto getFieldOffset() const {
      auto &propStorage = this->fieldOffset;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setFieldOffset(const ::mlir::IntegerAttr &propValue) {
      this->fieldOffset = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.fieldOffset == this->fieldOffset &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  SetFieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("jlcs.set_field", odsAttrs.getContext());
  }

  SetFieldOpGenericAdaptorBase(SetFieldOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getFieldOffsetAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().fieldOffset);
    return attr;
  }

  uint64_t getFieldOffset();
};
} // namespace detail
template <typename RangeT>
class SetFieldOpGenericAdaptor : public detail::SetFieldOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SetFieldOpGenericAdaptorBase;
  static constexpr int odsIndex_structValue = 0;
  static constexpr int odsIndex_newValue = 1;
public:
  SetFieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  SetFieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : SetFieldOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  SetFieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : SetFieldOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  SetFieldOpGenericAdaptor(RangeT values, const SetFieldOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = SetFieldOp, typename = std::enable_if_t<std::is_same_v<LateInst, SetFieldOp>>>
  SetFieldOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getStructValue() {
    return (*getODSOperands(0).begin());
  }

  ValueT getNewValue() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SetFieldOpAdaptor : public SetFieldOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SetFieldOpGenericAdaptor::SetFieldOpGenericAdaptor;
  SetFieldOpAdaptor(SetFieldOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class SetFieldOp : public ::mlir::Op<SetFieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SetFieldOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SetFieldOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_structValue = 0;
  static constexpr int odsIndex_newValue = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("fieldOffset")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getFieldOffsetAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getFieldOffsetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("jlcs.set_field");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getStructValue() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getNewValue() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getStructValueMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getNewValueMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getFieldOffsetAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().fieldOffset);
  }

  uint64_t getFieldOffset();
  void setFieldOffsetAttr(::mlir::IntegerAttr attr) {
    getProperties().fieldOffset = attr;
  }

  void setFieldOffset(uint64_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value structValue, Value newValue, IntegerAttr fieldOffset);
  static SetFieldOp create(::mlir::OpBuilder &builder, ::mlir::Location location, Value structValue, Value newValue, IntegerAttr fieldOffset);
  static SetFieldOp create(::mlir::ImplicitLocOpBuilder &builder, Value structValue, Value newValue, IntegerAttr fieldOffset);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace jlcs
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::jlcs::SetFieldOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::StoreArrayElementOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class StoreArrayElementOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  StoreArrayElementOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("jlcs.store_array_element", odsAttrs.getContext());
  }

  StoreArrayElementOpGenericAdaptorBase(StoreArrayElementOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};
} // namespace detail
template <typename RangeT>
class StoreArrayElementOpGenericAdaptor : public detail::StoreArrayElementOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::StoreArrayElementOpGenericAdaptorBase;
  static constexpr int odsIndex_value = 0;
  static constexpr int odsIndex_view = 1;
  static constexpr int odsIndex_indices = 2;
public:
  StoreArrayElementOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  StoreArrayElementOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : StoreArrayElementOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  StoreArrayElementOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : StoreArrayElementOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  StoreArrayElementOpGenericAdaptor(RangeT values, const StoreArrayElementOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = StoreArrayElementOp, typename = std::enable_if_t<std::is_same_v<LateInst, StoreArrayElementOp>>>
  StoreArrayElementOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getValue() {
    return (*getODSOperands(0).begin());
  }

  ValueT getView() {
    return (*getODSOperands(1).begin());
  }

  RangeT getIndices() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class StoreArrayElementOpAdaptor : public StoreArrayElementOpGenericAdaptor<::mlir::ValueRange> {
public:
  using StoreArrayElementOpGenericAdaptor::StoreArrayElementOpGenericAdaptor;
  StoreArrayElementOpAdaptor(StoreArrayElementOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class StoreArrayElementOp : public ::mlir::Op<StoreArrayElementOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StoreArrayElementOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = StoreArrayElementOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_value = 0;
  static constexpr int odsIndex_view = 1;
  static constexpr int odsIndex_indices = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("jlcs.store_array_element");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getValue() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getView() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::Operation::operand_range getIndices() {
    return getODSOperands(2);
  }

  ::mlir::OpOperand &getValueMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getViewMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value, Value view, ValueRange indices);
  static StoreArrayElementOp create(::mlir::OpBuilder &builder, ::mlir::Location location, Value value, Value view, ValueRange indices);
  static StoreArrayElementOp create(::mlir::ImplicitLocOpBuilder &builder, Value value, Value view, ValueRange indices);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};
} // namespace jlcs
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::jlcs::StoreArrayElementOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::TypeInfoOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TypeInfoOpGenericAdaptorBase {
public:
  struct Properties {
    using structTypeTy = ::mlir::TypeAttr;
    structTypeTy structType;

    auto getStructType() const {
      auto &propStorage = this->structType;
      return ::llvm::cast<::mlir::TypeAttr>(propStorage);
    }
    void setStructType(const ::mlir::TypeAttr &propValue) {
      this->structType = propValue;
    }
    using superTypeTy = ::mlir::StringAttr;
    superTypeTy superType;

    auto getSuperType() const {
      auto &propStorage = this->superType;
      return ::llvm::dyn_cast_or_null<::mlir::StringAttr>(propStorage);
    }
    void setSuperType(const ::mlir::StringAttr &propValue) {
      this->superType = propValue;
    }
    using typeNameTy = ::mlir::StringAttr;
    typeNameTy typeName;

    auto getTypeName() const {
      auto &propStorage = this->typeName;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setTypeName(const ::mlir::StringAttr &propValue) {
      this->typeName = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.structType == this->structType &&
        rhs.superType == this->superType &&
        rhs.typeName == this->typeName &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  TypeInfoOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("jlcs.type_info", odsAttrs.getContext());
  }

  TypeInfoOpGenericAdaptorBase(TypeInfoOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getTypeNameAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().typeName);
    return attr;
  }

  ::llvm::StringRef getTypeName();
  ::mlir::TypeAttr getStructTypeAttr() {
    auto attr = ::llvm::cast<::mlir::TypeAttr>(getProperties().structType);
    return attr;
  }

  ::mlir::Type getStructType();
  ::mlir::StringAttr getSuperTypeAttr();
  ::llvm::StringRef getSuperType();
};
} // namespace detail
template <typename RangeT>
class TypeInfoOpGenericAdaptor : public detail::TypeInfoOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TypeInfoOpGenericAdaptorBase;
public:
  TypeInfoOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TypeInfoOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TypeInfoOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  TypeInfoOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : TypeInfoOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  TypeInfoOpGenericAdaptor(RangeT values, const TypeInfoOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TypeInfoOp, typename = std::enable_if_t<std::is_same_v<LateInst, TypeInfoOp>>>
  TypeInfoOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TypeInfoOpAdaptor : public TypeInfoOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TypeInfoOpGenericAdaptor::TypeInfoOpGenericAdaptor;
  TypeInfoOpAdaptor(TypeInfoOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TypeInfoOp : public ::mlir::Op<TypeInfoOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsIsolatedFromAbove> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TypeInfoOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TypeInfoOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("structType"), ::llvm::StringRef("superType"), ::llvm::StringRef("typeName")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getStructTypeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getStructTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSuperTypeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSuperTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getTypeNameAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getTypeNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("jlcs.type_info");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::StringAttr getTypeNameAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().typeName);
  }

  ::llvm::StringRef getTypeName();
  ::mlir::TypeAttr getStructTypeAttr() {
    return ::llvm::cast<::mlir::TypeAttr>(getProperties().structType);
  }

  ::mlir::Type getStructType();
  ::mlir::StringAttr getSuperTypeAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::StringAttr>(getProperties().superType);
  }

  ::llvm::StringRef getSuperType();
  void setTypeNameAttr(::mlir::StringAttr attr) {
    getProperties().typeName = attr;
  }

  void setTypeName(::llvm::StringRef attrValue);
  void setStructTypeAttr(::mlir::TypeAttr attr) {
    getProperties().structType = attr;
  }

  void setStructType(::mlir::Type attrValue);
  void setSuperTypeAttr(::mlir::StringAttr attr) {
    getProperties().superType = attr;
  }

  void setSuperType(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringAttr typeName, TypeAttr structType, StringAttr superType);
  static TypeInfoOp create(::mlir::OpBuilder &builder, ::mlir::Location location, StringAttr typeName, TypeAttr structType, StringAttr superType);
  static TypeInfoOp create(::mlir::ImplicitLocOpBuilder &builder, StringAttr typeName, TypeAttr structType, StringAttr superType);
  static void populateDefaultProperties(::mlir::OperationName opName, Properties &properties);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace jlcs
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::jlcs::TypeInfoOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::VirtualCallOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class VirtualCallOpGenericAdaptorBase {
public:
  struct Properties {
    using class_nameTy = ::mlir::SymbolRefAttr;
    class_nameTy class_name;

    auto getClassName() const {
      auto &propStorage = this->class_name;
      return ::llvm::cast<::mlir::SymbolRefAttr>(propStorage);
    }
    void setClassName(const ::mlir::SymbolRefAttr &propValue) {
      this->class_name = propValue;
    }
    using slotTy = ::mlir::IntegerAttr;
    slotTy slot;

    auto getSlot() const {
      auto &propStorage = this->slot;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setSlot(const ::mlir::IntegerAttr &propValue) {
      this->slot = propValue;
    }
    using vtable_offsetTy = ::mlir::IntegerAttr;
    vtable_offsetTy vtable_offset;

    auto getVtableOffset() const {
      auto &propStorage = this->vtable_offset;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setVtableOffset(const ::mlir::IntegerAttr &propValue) {
      this->vtable_offset = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.class_name == this->class_name &&
        rhs.slot == this->slot &&
        rhs.vtable_offset == this->vtable_offset &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  VirtualCallOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("jlcs.vcall", odsAttrs.getContext());
  }

  VirtualCallOpGenericAdaptorBase(VirtualCallOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::SymbolRefAttr getClassNameAttr() {
    auto attr = ::llvm::cast<::mlir::SymbolRefAttr>(getProperties().class_name);
    return attr;
  }

  ::mlir::SymbolRefAttr getClassName();
  ::mlir::IntegerAttr getVtableOffsetAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().vtable_offset);
    return attr;
  }

  uint64_t getVtableOffset();
  ::mlir::IntegerAttr getSlotAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().slot);
    return attr;
  }

  uint64_t getSlot();
};
} // namespace detail
template <typename RangeT>
class VirtualCallOpGenericAdaptor : public detail::VirtualCallOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::VirtualCallOpGenericAdaptorBase;
  static constexpr int odsIndex_args = 0;
public:
  VirtualCallOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  VirtualCallOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : VirtualCallOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  VirtualCallOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : VirtualCallOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  VirtualCallOpGenericAdaptor(RangeT values, const VirtualCallOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = VirtualCallOp, typename = std::enable_if_t<std::is_same_v<LateInst, VirtualCallOp>>>
  VirtualCallOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getArgs() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class VirtualCallOpAdaptor : public VirtualCallOpGenericAdaptor<::mlir::ValueRange> {
public:
  using VirtualCallOpGenericAdaptor::VirtualCallOpGenericAdaptor;
  VirtualCallOpAdaptor(VirtualCallOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class VirtualCallOp : public ::mlir::Op<VirtualCallOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = VirtualCallOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = VirtualCallOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_args = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("class_name"), ::llvm::StringRef("slot"), ::llvm::StringRef("vtable_offset")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getClassNameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getClassNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSlotAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSlotAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getVtableOffsetAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getVtableOffsetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("jlcs.vcall");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getArgs() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getResult() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::Type>() : ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*results.begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::SymbolRefAttr getClassNameAttr() {
    return ::llvm::cast<::mlir::SymbolRefAttr>(getProperties().class_name);
  }

  ::mlir::SymbolRefAttr getClassName();
  ::mlir::IntegerAttr getVtableOffsetAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().vtable_offset);
  }

  uint64_t getVtableOffset();
  ::mlir::IntegerAttr getSlotAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().slot);
  }

  uint64_t getSlot();
  void setClassNameAttr(::mlir::SymbolRefAttr attr) {
    getProperties().class_name = attr;
  }

  void setVtableOffsetAttr(::mlir::IntegerAttr attr) {
    getProperties().vtable_offset = attr;
  }

  void setVtableOffset(uint64_t attrValue);
  void setSlotAttr(::mlir::IntegerAttr attr) {
    getProperties().slot = attr;
  }

  void setSlot(uint64_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, SymbolRefAttr class_name, ValueRange args, IntegerAttr vtable_offset, IntegerAttr slot, Type resultType);
  static VirtualCallOp create(::mlir::OpBuilder &builder, ::mlir::Location location, SymbolRefAttr class_name, ValueRange args, IntegerAttr vtable_offset, IntegerAttr slot, Type resultType);
  static VirtualCallOp create(::mlir::ImplicitLocOpBuilder &builder, SymbolRefAttr class_name, ValueRange args, IntegerAttr vtable_offset, IntegerAttr slot, Type resultType);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace jlcs
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::jlcs::VirtualCallOp)


#endif  // GET_OP_CLASSES

