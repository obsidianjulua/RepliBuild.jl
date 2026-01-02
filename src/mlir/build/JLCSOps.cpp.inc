/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: JLCS.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::jlcs::GetFieldOp,
::mlir::jlcs::FFECallOp,
::mlir::jlcs::LoadArrayElementOp,
::mlir::jlcs::SetFieldOp,
::mlir::jlcs::StoreArrayElementOp,
::mlir::jlcs::TypeInfoOp,
::mlir::jlcs::VirtualCallOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace jlcs {

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_JLCS1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_JLCS2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of any type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_JLCS3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::IndexType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of index, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_JLCS1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_JLCS2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::SymbolRefAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: symbol reference attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_JLCS3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
} // namespace jlcs
} // namespace mlir
namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::GetFieldOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetFieldOpGenericAdaptorBase::GetFieldOpGenericAdaptorBase(GetFieldOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint64_t GetFieldOpGenericAdaptorBase::getFieldOffset() {
  auto attr = getFieldOffsetAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
GetFieldOpAdaptor::GetFieldOpAdaptor(GetFieldOp op) : GetFieldOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult GetFieldOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_fieldOffset = getProperties().fieldOffset; (void)tblgen_fieldOffset;
  if (!tblgen_fieldOffset) return emitError(loc, "'jlcs.get_field' op ""requires attribute 'fieldOffset'");

  if (tblgen_fieldOffset && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_fieldOffset))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_fieldOffset).getType().isSignlessInteger(64)))))
    return emitError(loc, "'jlcs.get_field' op ""attribute 'fieldOffset' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult GetFieldOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.fieldOffset;
       auto attr = dict.get("fieldOffset");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `fieldOffset` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GetFieldOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.fieldOffset;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("fieldOffset",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code GetFieldOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.fieldOffset.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GetFieldOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "fieldOffset")
      return prop.fieldOffset;
  return std::nullopt;
}

void GetFieldOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "fieldOffset") {
       prop.fieldOffset = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.fieldOffset)>>(value);
       return;
    }
}

void GetFieldOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.fieldOffset) attrs.append("fieldOffset", prop.fieldOffset);
}

::llvm::LogicalResult GetFieldOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getFieldOffsetAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(attr, "fieldOffset", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult GetFieldOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.fieldOffset)))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetFieldOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.fieldOffset);
}

uint64_t GetFieldOp::getFieldOffset() {
  auto attr = getFieldOffsetAttr();
  return attr.getValue().getZExtValue();
}

void GetFieldOp::setFieldOffset(uint64_t attrValue) {
  getProperties().fieldOffset = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

GetFieldOp GetFieldOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, Value structValue, IntegerAttr fieldOffset, Type resultType) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, structValue, fieldOffset, resultType);
  auto __res__ = ::llvm::dyn_cast<GetFieldOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

GetFieldOp GetFieldOp::create(::mlir::ImplicitLocOpBuilder &builder, Value structValue, IntegerAttr fieldOffset, Type resultType) {
  return create(builder, builder.getLoc(), structValue, fieldOffset, resultType);
}

::llvm::LogicalResult GetFieldOp::verifyInvariantsImpl() {
  auto tblgen_fieldOffset = getProperties().fieldOffset; (void)tblgen_fieldOffset;
  if (!tblgen_fieldOffset) return emitOpError("requires attribute 'fieldOffset'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(*this, tblgen_fieldOffset, "fieldOffset")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult GetFieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult GetFieldOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;
  {

    auto &propStorage = prop.fieldOffset;
    auto fieldOffsetAttrName = ::mlir::StringAttr::get(ctx, "fieldOffset");
    auto attr = dict.get(fieldOffsetAttrName);
    usedKeys.insert(fieldOffsetAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for fieldOffset in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `fieldOffset` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::GetFieldOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::FFECallOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FFECallOpGenericAdaptorBase::FFECallOpGenericAdaptorBase(FFECallOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> FFECallOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
FFECallOpAdaptor::FFECallOpAdaptor(FFECallOp op) : FFECallOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult FFECallOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FFECallOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange FFECallOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FFECallOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::llvm::LogicalResult FFECallOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute FFECallOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code FFECallOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> FFECallOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void FFECallOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void FFECallOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult FFECallOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void FFECallOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

FFECallOp FFECallOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, operands, attributes);
  auto __res__ = ::llvm::dyn_cast<FFECallOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

FFECallOp FFECallOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), resultTypes, operands, attributes);
}

void FFECallOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  odsState.addTypes(resultTypes);
}

FFECallOp FFECallOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, operands, properties, discardableAttributes);
  auto __res__ = ::llvm::dyn_cast<FFECallOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

FFECallOp FFECallOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), resultTypes, operands, properties, discardableAttributes);
}

::llvm::LogicalResult FFECallOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult FFECallOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FFECallOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FFECallOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArgs();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getArgs().getTypes();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getResults().getTypes();
}

void FFECallOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::FFECallOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::LoadArrayElementOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LoadArrayElementOpGenericAdaptorBase::LoadArrayElementOpGenericAdaptorBase(LoadArrayElementOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> LoadArrayElementOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
LoadArrayElementOpAdaptor::LoadArrayElementOpAdaptor(LoadArrayElementOp op) : LoadArrayElementOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoadArrayElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoadArrayElementOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange LoadArrayElementOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult LoadArrayElementOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute LoadArrayElementOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code LoadArrayElementOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> LoadArrayElementOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void LoadArrayElementOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void LoadArrayElementOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult LoadArrayElementOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

LoadArrayElementOp LoadArrayElementOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, Value view, ValueRange indices, Type resultType) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, view, indices, resultType);
  auto __res__ = ::llvm::dyn_cast<LoadArrayElementOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

LoadArrayElementOp LoadArrayElementOp::create(::mlir::ImplicitLocOpBuilder &builder, Value view, ValueRange indices, Type resultType) {
  return create(builder, builder.getLoc(), view, indices, resultType);
}

::llvm::LogicalResult LoadArrayElementOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoadArrayElementOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult LoadArrayElementOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::LoadArrayElementOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::SetFieldOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SetFieldOpGenericAdaptorBase::SetFieldOpGenericAdaptorBase(SetFieldOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint64_t SetFieldOpGenericAdaptorBase::getFieldOffset() {
  auto attr = getFieldOffsetAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
SetFieldOpAdaptor::SetFieldOpAdaptor(SetFieldOp op) : SetFieldOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SetFieldOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_fieldOffset = getProperties().fieldOffset; (void)tblgen_fieldOffset;
  if (!tblgen_fieldOffset) return emitError(loc, "'jlcs.set_field' op ""requires attribute 'fieldOffset'");

  if (tblgen_fieldOffset && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_fieldOffset))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_fieldOffset).getType().isSignlessInteger(64)))))
    return emitError(loc, "'jlcs.set_field' op ""attribute 'fieldOffset' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult SetFieldOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.fieldOffset;
       auto attr = dict.get("fieldOffset");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `fieldOffset` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SetFieldOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.fieldOffset;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("fieldOffset",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code SetFieldOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.fieldOffset.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SetFieldOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "fieldOffset")
      return prop.fieldOffset;
  return std::nullopt;
}

void SetFieldOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "fieldOffset") {
       prop.fieldOffset = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.fieldOffset)>>(value);
       return;
    }
}

void SetFieldOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.fieldOffset) attrs.append("fieldOffset", prop.fieldOffset);
}

::llvm::LogicalResult SetFieldOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getFieldOffsetAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(attr, "fieldOffset", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult SetFieldOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.fieldOffset)))
    return ::mlir::failure();
  return ::mlir::success();
}

void SetFieldOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.fieldOffset);
}

uint64_t SetFieldOp::getFieldOffset() {
  auto attr = getFieldOffsetAttr();
  return attr.getValue().getZExtValue();
}

void SetFieldOp::setFieldOffset(uint64_t attrValue) {
  getProperties().fieldOffset = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

SetFieldOp SetFieldOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, Value structValue, Value newValue, IntegerAttr fieldOffset) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, structValue, newValue, fieldOffset);
  auto __res__ = ::llvm::dyn_cast<SetFieldOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

SetFieldOp SetFieldOp::create(::mlir::ImplicitLocOpBuilder &builder, Value structValue, Value newValue, IntegerAttr fieldOffset) {
  return create(builder, builder.getLoc(), structValue, newValue, fieldOffset);
}

::llvm::LogicalResult SetFieldOp::verifyInvariantsImpl() {
  auto tblgen_fieldOffset = getProperties().fieldOffset; (void)tblgen_fieldOffset;
  if (!tblgen_fieldOffset) return emitOpError("requires attribute 'fieldOffset'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(*this, tblgen_fieldOffset, "fieldOffset")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult SetFieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult SetFieldOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;
  {

    auto &propStorage = prop.fieldOffset;
    auto fieldOffsetAttrName = ::mlir::StringAttr::get(ctx, "fieldOffset");
    auto attr = dict.get(fieldOffsetAttrName);
    usedKeys.insert(fieldOffsetAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for fieldOffset in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `fieldOffset` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void SetFieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::SetFieldOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::StoreArrayElementOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StoreArrayElementOpGenericAdaptorBase::StoreArrayElementOpGenericAdaptorBase(StoreArrayElementOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StoreArrayElementOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
StoreArrayElementOpAdaptor::StoreArrayElementOpAdaptor(StoreArrayElementOp op) : StoreArrayElementOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult StoreArrayElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StoreArrayElementOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange StoreArrayElementOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult StoreArrayElementOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute StoreArrayElementOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StoreArrayElementOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> StoreArrayElementOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void StoreArrayElementOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void StoreArrayElementOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult StoreArrayElementOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

StoreArrayElementOp StoreArrayElementOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, Value value, Value view, ValueRange indices) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, value, view, indices);
  auto __res__ = ::llvm::dyn_cast<StoreArrayElementOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

StoreArrayElementOp StoreArrayElementOp::create(::mlir::ImplicitLocOpBuilder &builder, Value value, Value view, ValueRange indices) {
  return create(builder, builder.getLoc(), value, view, indices);
}

::llvm::LogicalResult StoreArrayElementOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult StoreArrayElementOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult StoreArrayElementOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::StoreArrayElementOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::TypeInfoOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TypeInfoOpGenericAdaptorBase::TypeInfoOpGenericAdaptorBase(TypeInfoOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef TypeInfoOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

uint64_t TypeInfoOpGenericAdaptorBase::getSize() {
  auto attr = getSizeAttr();
  return attr.getValue().getZExtValue();
}

uint64_t TypeInfoOpGenericAdaptorBase::getVtableOffset() {
  auto attr = getVtableOffsetAttr();
  return attr.getValue().getZExtValue();
}

uint64_t TypeInfoOpGenericAdaptorBase::getVtableAddr() {
  auto attr = getVtableAddrAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
TypeInfoOpAdaptor::TypeInfoOpAdaptor(TypeInfoOp op) : TypeInfoOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TypeInfoOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_size = getProperties().size; (void)tblgen_size;
  if (!tblgen_size) return emitError(loc, "'jlcs.type_info' op ""requires attribute 'size'");
  auto tblgen_sym_name = getProperties().sym_name; (void)tblgen_sym_name;
  if (!tblgen_sym_name) return emitError(loc, "'jlcs.type_info' op ""requires attribute 'sym_name'");
  auto tblgen_vtable_addr = getProperties().vtable_addr; (void)tblgen_vtable_addr;
  if (!tblgen_vtable_addr) return emitError(loc, "'jlcs.type_info' op ""requires attribute 'vtable_addr'");
  auto tblgen_vtable_offset = getProperties().vtable_offset; (void)tblgen_vtable_offset;
  if (!tblgen_vtable_offset) return emitError(loc, "'jlcs.type_info' op ""requires attribute 'vtable_offset'");

  if (tblgen_sym_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_sym_name))))
    return emitError(loc, "'jlcs.type_info' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_size && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_size))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_size).getType().isSignlessInteger(64)))))
    return emitError(loc, "'jlcs.type_info' op ""attribute 'size' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_vtable_offset && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_vtable_offset))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_vtable_offset).getType().isSignlessInteger(64)))))
    return emitError(loc, "'jlcs.type_info' op ""attribute 'vtable_offset' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_vtable_addr && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_vtable_addr))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_vtable_addr).getType().isSignlessInteger(64)))))
    return emitError(loc, "'jlcs.type_info' op ""attribute 'vtable_addr' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult TypeInfoOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.size;
       auto attr = dict.get("size");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `size` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.sym_name;
       auto attr = dict.get("sym_name");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `sym_name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.vtable_addr;
       auto attr = dict.get("vtable_addr");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `vtable_addr` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.vtable_offset;
       auto attr = dict.get("vtable_offset");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `vtable_offset` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TypeInfoOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.size;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("size",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.sym_name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("sym_name",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.vtable_addr;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("vtable_addr",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.vtable_offset;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("vtable_offset",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TypeInfoOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.size.getAsOpaquePointer()), 
    llvm::hash_value(prop.sym_name.getAsOpaquePointer()), 
    llvm::hash_value(prop.vtable_addr.getAsOpaquePointer()), 
    llvm::hash_value(prop.vtable_offset.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TypeInfoOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "size")
      return prop.size;

    if (name == "sym_name")
      return prop.sym_name;

    if (name == "vtable_addr")
      return prop.vtable_addr;

    if (name == "vtable_offset")
      return prop.vtable_offset;
  return std::nullopt;
}

void TypeInfoOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "size") {
       prop.size = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.size)>>(value);
       return;
    }

    if (name == "sym_name") {
       prop.sym_name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.sym_name)>>(value);
       return;
    }

    if (name == "vtable_addr") {
       prop.vtable_addr = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.vtable_addr)>>(value);
       return;
    }

    if (name == "vtable_offset") {
       prop.vtable_offset = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.vtable_offset)>>(value);
       return;
    }
}

void TypeInfoOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.size) attrs.append("size", prop.size);

    if (prop.sym_name) attrs.append("sym_name", prop.sym_name);

    if (prop.vtable_addr) attrs.append("vtable_addr", prop.vtable_addr);

    if (prop.vtable_offset) attrs.append("vtable_offset", prop.vtable_offset);
}

::llvm::LogicalResult TypeInfoOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getSizeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(attr, "size", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSymNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS2(attr, "sym_name", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getVtableAddrAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(attr, "vtable_addr", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getVtableOffsetAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(attr, "vtable_offset", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TypeInfoOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.size)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.sym_name)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.vtable_addr)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.vtable_offset)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TypeInfoOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.size);
  writer.writeAttribute(prop.sym_name);
  writer.writeAttribute(prop.vtable_addr);
  writer.writeAttribute(prop.vtable_offset);
}

::llvm::StringRef TypeInfoOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

uint64_t TypeInfoOp::getSize() {
  auto attr = getSizeAttr();
  return attr.getValue().getZExtValue();
}

uint64_t TypeInfoOp::getVtableOffset() {
  auto attr = getVtableOffsetAttr();
  return attr.getValue().getZExtValue();
}

uint64_t TypeInfoOp::getVtableAddr() {
  auto attr = getVtableAddrAttr();
  return attr.getValue().getZExtValue();
}

void TypeInfoOp::setSymName(::llvm::StringRef attrValue) {
  getProperties().sym_name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void TypeInfoOp::setSize(uint64_t attrValue) {
  getProperties().size = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TypeInfoOp::setVtableOffset(uint64_t attrValue) {
  getProperties().vtable_offset = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TypeInfoOp::setVtableAddr(uint64_t attrValue) {
  getProperties().vtable_addr = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TypeInfoOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr size, ::mlir::IntegerAttr vtable_offset, ::mlir::IntegerAttr vtable_addr) {
  odsState.getOrAddProperties<Properties>().sym_name = sym_name;
  odsState.getOrAddProperties<Properties>().size = size;
  odsState.getOrAddProperties<Properties>().vtable_offset = vtable_offset;
  odsState.getOrAddProperties<Properties>().vtable_addr = vtable_addr;
}

TypeInfoOp TypeInfoOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr size, ::mlir::IntegerAttr vtable_offset, ::mlir::IntegerAttr vtable_addr) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, sym_name, size, vtable_offset, vtable_addr);
  auto __res__ = ::llvm::dyn_cast<TypeInfoOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TypeInfoOp TypeInfoOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr size, ::mlir::IntegerAttr vtable_offset, ::mlir::IntegerAttr vtable_addr) {
  return create(builder, builder.getLoc(), sym_name, size, vtable_offset, vtable_addr);
}

void TypeInfoOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr size, ::mlir::IntegerAttr vtable_offset, ::mlir::IntegerAttr vtable_addr) {
  odsState.getOrAddProperties<Properties>().sym_name = sym_name;
  odsState.getOrAddProperties<Properties>().size = size;
  odsState.getOrAddProperties<Properties>().vtable_offset = vtable_offset;
  odsState.getOrAddProperties<Properties>().vtable_addr = vtable_addr;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

TypeInfoOp TypeInfoOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr size, ::mlir::IntegerAttr vtable_offset, ::mlir::IntegerAttr vtable_addr) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, sym_name, size, vtable_offset, vtable_addr);
  auto __res__ = ::llvm::dyn_cast<TypeInfoOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TypeInfoOp TypeInfoOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr size, ::mlir::IntegerAttr vtable_offset, ::mlir::IntegerAttr vtable_addr) {
  return create(builder, builder.getLoc(), resultTypes, sym_name, size, vtable_offset, vtable_addr);
}

void TypeInfoOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, uint64_t size, uint64_t vtable_offset, uint64_t vtable_addr) {
  odsState.getOrAddProperties<Properties>().sym_name = odsBuilder.getStringAttr(sym_name);
  odsState.getOrAddProperties<Properties>().size = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), size);
  odsState.getOrAddProperties<Properties>().vtable_offset = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), vtable_offset);
  odsState.getOrAddProperties<Properties>().vtable_addr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), vtable_addr);
}

TypeInfoOp TypeInfoOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::llvm::StringRef sym_name, uint64_t size, uint64_t vtable_offset, uint64_t vtable_addr) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, sym_name, size, vtable_offset, vtable_addr);
  auto __res__ = ::llvm::dyn_cast<TypeInfoOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TypeInfoOp TypeInfoOp::create(::mlir::ImplicitLocOpBuilder &builder, ::llvm::StringRef sym_name, uint64_t size, uint64_t vtable_offset, uint64_t vtable_addr) {
  return create(builder, builder.getLoc(), sym_name, size, vtable_offset, vtable_addr);
}

void TypeInfoOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, uint64_t size, uint64_t vtable_offset, uint64_t vtable_addr) {
  odsState.getOrAddProperties<Properties>().sym_name = odsBuilder.getStringAttr(sym_name);
  odsState.getOrAddProperties<Properties>().size = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), size);
  odsState.getOrAddProperties<Properties>().vtable_offset = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), vtable_offset);
  odsState.getOrAddProperties<Properties>().vtable_addr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), vtable_addr);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

TypeInfoOp TypeInfoOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, uint64_t size, uint64_t vtable_offset, uint64_t vtable_addr) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, sym_name, size, vtable_offset, vtable_addr);
  auto __res__ = ::llvm::dyn_cast<TypeInfoOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TypeInfoOp TypeInfoOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, uint64_t size, uint64_t vtable_offset, uint64_t vtable_addr) {
  return create(builder, builder.getLoc(), resultTypes, sym_name, size, vtable_offset, vtable_addr);
}

void TypeInfoOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TypeInfoOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

TypeInfoOp TypeInfoOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, operands, attributes);
  auto __res__ = ::llvm::dyn_cast<TypeInfoOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TypeInfoOp TypeInfoOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), resultTypes, operands, attributes);
}

void TypeInfoOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

TypeInfoOp TypeInfoOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, operands, properties, discardableAttributes);
  auto __res__ = ::llvm::dyn_cast<TypeInfoOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TypeInfoOp TypeInfoOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), resultTypes, operands, properties, discardableAttributes);
}

::llvm::LogicalResult TypeInfoOp::verifyInvariantsImpl() {
  auto tblgen_size = getProperties().size; (void)tblgen_size;
  if (!tblgen_size) return emitOpError("requires attribute 'size'");
  auto tblgen_sym_name = getProperties().sym_name; (void)tblgen_sym_name;
  if (!tblgen_sym_name) return emitOpError("requires attribute 'sym_name'");
  auto tblgen_vtable_addr = getProperties().vtable_addr; (void)tblgen_vtable_addr;
  if (!tblgen_vtable_addr) return emitOpError("requires attribute 'vtable_addr'");
  auto tblgen_vtable_offset = getProperties().vtable_offset; (void)tblgen_vtable_offset;
  if (!tblgen_vtable_offset) return emitOpError("requires attribute 'vtable_offset'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS2(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(*this, tblgen_size, "size")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(*this, tblgen_vtable_offset, "vtable_offset")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(*this, tblgen_vtable_addr, "vtable_addr")))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::LogicalResult TypeInfoOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TypeInfoOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;

  if (parser.parseSymbolName(sym_nameAttr))
    return ::mlir::failure();
  if (sym_nameAttr) result.getOrAddProperties<TypeInfoOp::Properties>().sym_name = sym_nameAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void TypeInfoOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("sym_name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::TypeInfoOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::VirtualCallOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
VirtualCallOpGenericAdaptorBase::VirtualCallOpGenericAdaptorBase(VirtualCallOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> VirtualCallOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::SymbolRefAttr VirtualCallOpGenericAdaptorBase::getClassName() {
  auto attr = getClassNameAttr();
  return attr;
}

uint64_t VirtualCallOpGenericAdaptorBase::getVtableOffset() {
  auto attr = getVtableOffsetAttr();
  return attr.getValue().getZExtValue();
}

uint64_t VirtualCallOpGenericAdaptorBase::getSlot() {
  auto attr = getSlotAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
VirtualCallOpAdaptor::VirtualCallOpAdaptor(VirtualCallOp op) : VirtualCallOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult VirtualCallOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_class_name = getProperties().class_name; (void)tblgen_class_name;
  if (!tblgen_class_name) return emitError(loc, "'jlcs.vcall' op ""requires attribute 'class_name'");
  auto tblgen_slot = getProperties().slot; (void)tblgen_slot;
  if (!tblgen_slot) return emitError(loc, "'jlcs.vcall' op ""requires attribute 'slot'");
  auto tblgen_vtable_offset = getProperties().vtable_offset; (void)tblgen_vtable_offset;
  if (!tblgen_vtable_offset) return emitError(loc, "'jlcs.vcall' op ""requires attribute 'vtable_offset'");

  if (tblgen_class_name && !((::llvm::isa<::mlir::SymbolRefAttr>(tblgen_class_name))))
    return emitError(loc, "'jlcs.vcall' op ""attribute 'class_name' failed to satisfy constraint: symbol reference attribute");

  if (tblgen_vtable_offset && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_vtable_offset))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_vtable_offset).getType().isSignlessInteger(64)))))
    return emitError(loc, "'jlcs.vcall' op ""attribute 'vtable_offset' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_slot && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_slot))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_slot).getType().isSignlessInteger(64)))))
    return emitError(loc, "'jlcs.vcall' op ""attribute 'slot' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VirtualCallOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange VirtualCallOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VirtualCallOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::llvm::LogicalResult VirtualCallOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.class_name;
       auto attr = dict.get("class_name");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `class_name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.slot;
       auto attr = dict.get("slot");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `slot` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.vtable_offset;
       auto attr = dict.get("vtable_offset");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `vtable_offset` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute VirtualCallOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.class_name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("class_name",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.slot;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("slot",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.vtable_offset;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("vtable_offset",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code VirtualCallOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.class_name.getAsOpaquePointer()), 
    llvm::hash_value(prop.slot.getAsOpaquePointer()), 
    llvm::hash_value(prop.vtable_offset.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> VirtualCallOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "class_name")
      return prop.class_name;

    if (name == "slot")
      return prop.slot;

    if (name == "vtable_offset")
      return prop.vtable_offset;
  return std::nullopt;
}

void VirtualCallOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "class_name") {
       prop.class_name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.class_name)>>(value);
       return;
    }

    if (name == "slot") {
       prop.slot = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.slot)>>(value);
       return;
    }

    if (name == "vtable_offset") {
       prop.vtable_offset = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.vtable_offset)>>(value);
       return;
    }
}

void VirtualCallOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.class_name) attrs.append("class_name", prop.class_name);

    if (prop.slot) attrs.append("slot", prop.slot);

    if (prop.vtable_offset) attrs.append("vtable_offset", prop.vtable_offset);
}

::llvm::LogicalResult VirtualCallOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getClassNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS3(attr, "class_name", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSlotAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(attr, "slot", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getVtableOffsetAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(attr, "vtable_offset", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult VirtualCallOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.class_name)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.slot)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.vtable_offset)))
    return ::mlir::failure();
  return ::mlir::success();
}

void VirtualCallOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.class_name);
  writer.writeAttribute(prop.slot);
  writer.writeAttribute(prop.vtable_offset);
}

::mlir::SymbolRefAttr VirtualCallOp::getClassName() {
  auto attr = getClassNameAttr();
  return attr;
}

uint64_t VirtualCallOp::getVtableOffset() {
  auto attr = getVtableOffsetAttr();
  return attr.getValue().getZExtValue();
}

uint64_t VirtualCallOp::getSlot() {
  auto attr = getSlotAttr();
  return attr.getValue().getZExtValue();
}

void VirtualCallOp::setVtableOffset(uint64_t attrValue) {
  getProperties().vtable_offset = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void VirtualCallOp::setSlot(uint64_t attrValue) {
  getProperties().slot = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

VirtualCallOp VirtualCallOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, SymbolRefAttr class_name, ValueRange args, IntegerAttr vtable_offset, IntegerAttr slot, Type resultType) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, class_name, args, vtable_offset, slot, resultType);
  auto __res__ = ::llvm::dyn_cast<VirtualCallOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

VirtualCallOp VirtualCallOp::create(::mlir::ImplicitLocOpBuilder &builder, SymbolRefAttr class_name, ValueRange args, IntegerAttr vtable_offset, IntegerAttr slot, Type resultType) {
  return create(builder, builder.getLoc(), class_name, args, vtable_offset, slot, resultType);
}

::llvm::LogicalResult VirtualCallOp::verifyInvariantsImpl() {
  auto tblgen_class_name = getProperties().class_name; (void)tblgen_class_name;
  if (!tblgen_class_name) return emitOpError("requires attribute 'class_name'");
  auto tblgen_slot = getProperties().slot; (void)tblgen_slot;
  if (!tblgen_slot) return emitOpError("requires attribute 'slot'");
  auto tblgen_vtable_offset = getProperties().vtable_offset; (void)tblgen_vtable_offset;
  if (!tblgen_vtable_offset) return emitOpError("requires attribute 'vtable_offset'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS3(*this, tblgen_class_name, "class_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(*this, tblgen_vtable_offset, "vtable_offset")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(*this, tblgen_slot, "slot")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("result group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult VirtualCallOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult VirtualCallOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;
  {

    auto &propStorage = prop.class_name;
    auto class_nameAttrName = ::mlir::StringAttr::get(ctx, "class_name");
    auto attr = dict.get(class_nameAttrName);
    usedKeys.insert(class_nameAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for class_name in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `class_name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  {

    auto &propStorage = prop.vtable_offset;
    auto vtable_offsetAttrName = ::mlir::StringAttr::get(ctx, "vtable_offset");
    auto attr = dict.get(vtable_offsetAttrName);
    usedKeys.insert(vtable_offsetAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for vtable_offset in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `vtable_offset` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  {

    auto &propStorage = prop.slot;
    auto slotAttrName = ::mlir::StringAttr::get(ctx, "slot");
    auto attr = dict.get(slotAttrName);
    usedKeys.insert(slotAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for slot in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `slot` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::VirtualCallOp)


#endif  // GET_OP_CLASSES

