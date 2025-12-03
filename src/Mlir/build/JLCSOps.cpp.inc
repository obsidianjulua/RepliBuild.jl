/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: JLCSOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::jlcs::GetFieldOp,
::mlir::jlcs::SetFieldOp,
::mlir::jlcs::TypeInfoOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace jlcs {

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_JLCSOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCSOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCSOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_JLCSOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCSOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCSOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_JLCSOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCSOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: any type attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCSOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_JLCSOps3(attr, attrName, [op]() {
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
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCSOps1(attr, "fieldOffset", emitError)))
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCSOps1(*this, tblgen_fieldOffset, "fieldOffset")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCSOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCSOps1(*this, v.getType(), "result", index++)))
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
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCSOps1(attr, "fieldOffset", emitError)))
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCSOps1(*this, tblgen_fieldOffset, "fieldOffset")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCSOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCSOps1(*this, v.getType(), "operand", index++)))
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
// ::mlir::jlcs::TypeInfoOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TypeInfoOpGenericAdaptorBase::TypeInfoOpGenericAdaptorBase(TypeInfoOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef TypeInfoOpGenericAdaptorBase::getTypeName() {
  auto attr = getTypeNameAttr();
  return attr.getValue();
}

::mlir::Type TypeInfoOpGenericAdaptorBase::getStructType() {
  auto attr = getStructTypeAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

::mlir::StringAttr TypeInfoOpGenericAdaptorBase::getSuperTypeAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::StringAttr>(getProperties().superType);
  return attr;
}

::llvm::StringRef TypeInfoOpGenericAdaptorBase::getSuperType() {
  auto attr = getSuperTypeAttr();
  return attr.getValue();
}

} // namespace detail
TypeInfoOpAdaptor::TypeInfoOpAdaptor(TypeInfoOp op) : TypeInfoOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TypeInfoOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_structType = getProperties().structType; (void)tblgen_structType;
  if (!tblgen_structType) return emitError(loc, "'jlcs.type_info' op ""requires attribute 'structType'");
  auto tblgen_superType = getProperties().superType; (void)tblgen_superType;
  auto tblgen_typeName = getProperties().typeName; (void)tblgen_typeName;
  if (!tblgen_typeName) return emitError(loc, "'jlcs.type_info' op ""requires attribute 'typeName'");

  if (tblgen_typeName && !((::llvm::isa<::mlir::StringAttr>(tblgen_typeName))))
    return emitError(loc, "'jlcs.type_info' op ""attribute 'typeName' failed to satisfy constraint: string attribute");

  if (tblgen_structType && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_structType))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(tblgen_structType).getValue()))) && ((true))))
    return emitError(loc, "'jlcs.type_info' op ""attribute 'structType' failed to satisfy constraint: any type attribute");

  if (tblgen_superType && !((::llvm::isa<::mlir::StringAttr>(tblgen_superType))))
    return emitError(loc, "'jlcs.type_info' op ""attribute 'superType' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult TypeInfoOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.structType;
       auto attr = dict.get("structType");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `structType` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.superType;
       auto attr = dict.get("superType");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `superType` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.typeName;
       auto attr = dict.get("typeName");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `typeName` in property conversion: " << attr;
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
      const auto &propStorage = prop.structType;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("structType",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.superType;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("superType",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.typeName;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("typeName",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TypeInfoOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.structType.getAsOpaquePointer()), 
    llvm::hash_value(prop.superType.getAsOpaquePointer()), 
    llvm::hash_value(prop.typeName.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TypeInfoOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "structType")
      return prop.structType;

    if (name == "superType")
      return prop.superType;

    if (name == "typeName")
      return prop.typeName;
  return std::nullopt;
}

void TypeInfoOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "structType") {
       prop.structType = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.structType)>>(value);
       return;
    }

    if (name == "superType") {
       prop.superType = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.superType)>>(value);
       return;
    }

    if (name == "typeName") {
       prop.typeName = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.typeName)>>(value);
       return;
    }
}

void TypeInfoOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.structType) attrs.append("structType", prop.structType);

    if (prop.superType) attrs.append("superType", prop.superType);

    if (prop.typeName) attrs.append("typeName", prop.typeName);
}

::llvm::LogicalResult TypeInfoOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getStructTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCSOps3(attr, "structType", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSuperTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCSOps2(attr, "superType", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTypeNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCSOps2(attr, "typeName", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TypeInfoOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.structType)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.superType)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.typeName)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TypeInfoOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.structType);

  writer.writeOptionalAttribute(prop.superType);
  writer.writeAttribute(prop.typeName);
}

::llvm::StringRef TypeInfoOp::getTypeName() {
  auto attr = getTypeNameAttr();
  return attr.getValue();
}

::mlir::Type TypeInfoOp::getStructType() {
  auto attr = getStructTypeAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

::llvm::StringRef TypeInfoOp::getSuperType() {
  auto attr = getSuperTypeAttr();
  return attr.getValue();
}

void TypeInfoOp::setTypeName(::llvm::StringRef attrValue) {
  getProperties().typeName = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void TypeInfoOp::setStructType(::mlir::Type attrValue) {
  getProperties().structType = ::mlir::TypeAttr::get(attrValue);
}

void TypeInfoOp::setSuperType(::llvm::StringRef attrValue) {
  getProperties().superType = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

TypeInfoOp TypeInfoOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, StringAttr typeName, TypeAttr structType, StringAttr superType) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, typeName, structType, superType);
  auto __res__ = ::llvm::dyn_cast<TypeInfoOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TypeInfoOp TypeInfoOp::create(::mlir::ImplicitLocOpBuilder &builder, StringAttr typeName, TypeAttr structType, StringAttr superType) {
  return create(builder, builder.getLoc(), typeName, structType, superType);
}

void TypeInfoOp::populateDefaultProperties(::mlir::OperationName opName, Properties &properties) {
  ::mlir::Builder odsBuilder(opName.getContext());
  if (!properties.superType)
    properties.superType = odsBuilder.getStringAttr("""");
}

::llvm::LogicalResult TypeInfoOp::verifyInvariantsImpl() {
  auto tblgen_structType = getProperties().structType; (void)tblgen_structType;
  if (!tblgen_structType) return emitOpError("requires attribute 'structType'");
  auto tblgen_superType = getProperties().superType; (void)tblgen_superType;
  auto tblgen_typeName = getProperties().typeName; (void)tblgen_typeName;
  if (!tblgen_typeName) return emitOpError("requires attribute 'typeName'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCSOps2(*this, tblgen_typeName, "typeName")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCSOps3(*this, tblgen_structType, "structType")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCSOps2(*this, tblgen_superType, "superType")))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::LogicalResult TypeInfoOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult TypeInfoOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
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

    auto &propStorage = prop.typeName;
    auto typeNameAttrName = ::mlir::StringAttr::get(ctx, "typeName");
    auto attr = dict.get(typeNameAttrName);
    usedKeys.insert(typeNameAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for typeName in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `typeName` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  {

    auto &propStorage = prop.structType;
    auto structTypeAttrName = ::mlir::StringAttr::get(ctx, "structType");
    auto attr = dict.get(structTypeAttrName);
    usedKeys.insert(structTypeAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for structType in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `structType` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  {

    auto &propStorage = prop.superType;
    auto superTypeAttrName = ::mlir::StringAttr::get(ctx, "superType");
    auto attr = dict.get(superTypeAttrName);
    usedKeys.insert(superTypeAttrName);
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        emitError() << "expected key entry for superType in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `superType` in property conversion: " << attr;
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

void TypeInfoOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::TypeInfoOp)


#endif  // GET_OP_CLASSES

