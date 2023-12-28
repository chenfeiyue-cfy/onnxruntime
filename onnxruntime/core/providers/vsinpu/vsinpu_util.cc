/****************************************************************************
 *
 *    Copyright (c) 2023 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#include "vsinpu_util.h"

namespace onnxruntime {

template <typename T>
struct shared_array_deletor {
  void operator()(T const* ptr) { delete[] ptr; }
};
namespace vsi {
namespace npu {
namespace util {
tim::vx::DataType OnnxDtypeToTIMVXDtype(const int32_t dtype) {
  switch (dtype) {
    case onnx::TensorProto_DataType_FLOAT:
      return tim::vx::DataType::FLOAT32;
    case onnx::TensorProto_DataType_FLOAT16:
      return tim::vx::DataType::FLOAT16;
    case onnx::TensorProto_DataType_INT8:
      return tim::vx::DataType::INT8;
    case onnx::TensorProto_DataType_UINT8:
      return tim::vx::DataType::UINT8;
    case onnx::TensorProto_DataType_INT32:
      return tim::vx::DataType::INT32;
    case onnx::TensorProto_DataType_INT16:
      return tim::vx::DataType::INT16;
    case onnx::TensorProto_DataType_BOOL:
      return tim::vx::DataType::INT8;
    default:
      LOGS_DEFAULT(WARNING) << "Unsupported data type: " << dtype;
      break;
  }
  return tim::vx::DataType::FLOAT32;
}

tim::vx::DataType OnnxDtypeToTIMVXDtype(const ONNX_NAMESPACE::DataType type) {
  static const std::map<std::string, tim::vx::DataType> type_table = {
      {"tensor(float)", tim::vx::DataType::FLOAT32},
      {"tensor(float16)", tim::vx::DataType::FLOAT16},
      {"tensor(int8)", tim::vx::DataType::INT8},
      {"tensor(uint8)", tim::vx::DataType::UINT8},
      {"tensor(int32)", tim::vx::DataType::INT32},
      {"tensor(int16)", tim::vx::DataType::INT16},
      {"tensor(bool)", tim::vx::DataType::INT8},
  };
  auto search = type_table.find(*type);
  if (search != type_table.end()) {
    return search->second;
  }
  LOGS_DEFAULT(WARNING) << "Unsupported data type: " << *type;
  return tim::vx::DataType::FLOAT32;
}

tim::vx::ShapeType OnnxShapeToTIMVXShape(const onnxruntime::TensorShape ts) {
  tim::vx::ShapeType timvx_shape(ts.NumDimensions());
  if (ts.NumDimensions() == 0) {
    timvx_shape.push_back(1);
  } else {
    for (int i = 0; i < ts.NumDimensions(); i++) {
      timvx_shape[i] = ts.GetDims()[i];
    }
  }
  return timvx_shape;
}

std::string PrintNode(const onnxruntime::NodeArg& node_arg) {
  auto shape = node_arg.Shape();
  if (shape == nullptr || shape->dim_size() == 0) {
    return "<null>";
  }
  std::string s = node_arg.Name() + ":<";
  for (int i = 0; i < shape->dim_size(); i++) {
    auto dim = shape->dim(i);
    std::string s1;
    std::stringstream ss;
    ss << dim.dim_value();
    ss >> s1;
    s += s1;
    if (i < shape->dim_size() - 1) {
      s += ",";
    } else {
      s += ">";
    }
  }
  return s;
}

std::string PrintNode(const std::vector<int64_t> shape) {
  if (shape.size() == 0) {
    return "<null>";
  }
  std::string s = "<";
  for (std::size_t i = 0; i < shape.size(); i++) {
    auto dim = shape[i];
    std::string s1;
    std::stringstream ss;
    ss << dim;
    ss >> s1;
    s += s1;
    if (i < shape.size() - 1) {
      s += ",";
    } else {
      s += ">";
    }
  }
  return s;
}

size_t GetTensorElementSize(const ONNXTensorElementDataType type) {
  switch (type) {
    case onnx::TensorProto_DataType_INT64:
      return 8;
    case onnx::TensorProto_DataType_FLOAT:
    case onnx::TensorProto_DataType_INT32:
      return 4;
    case onnx::TensorProto_DataType_FLOAT16:
    case onnx::TensorProto_DataType_INT16:
    case onnx::TensorProto_DataType_UINT16:
      return 2;
    case onnx::TensorProto_DataType_INT8:
    case onnx::TensorProto_DataType_UINT8:
    case onnx::TensorProto_DataType_BOOL:
      return 1;
    default:
      break;
  }
  return 0;
}

size_t GetTensorBytes(const Ort::TensorTypeAndShapeInfo& info) {
  return info.GetElementCount() * GetTensorElementSize(info.GetElementType());
}

TensorShape GetTensorShape(const onnxruntime::NodeArg& node_arg) {
  auto shape_proto = node_arg.Shape();
  std::vector<int64_t> dims;
  if (shape_proto != nullptr) {
    for (int i = 0; i < shape_proto->dim_size(); i++) {
      auto dim = shape_proto->dim(i);
      dims.push_back(dim.dim_value());
    }
  }
  if (dims.size() == 0) {
    dims.push_back(1);
  }
  TensorShape ts(dims);
  return ts;
}

std::shared_ptr<uint8_t> UnpackTensor(
    const NodeArg* node_arg, const ONNX_NAMESPACE::TensorProto& initializer) {
  std::shared_ptr<uint8_t> unpackedTensor;
  auto shape = GetTensorShape(*node_arg);
  size_t elementCount = shape.Size();

#define CASE_PROTO(X, Y)                                                      \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X: {      \
    size_t tensorByteSize = elementCount * sizeof(Y);                         \
    unpackedTensor.reset(new uint8_t[tensorByteSize],                         \
                         shared_array_deletor<uint8_t>());                    \
    auto status = onnxruntime::utils::UnpackTensor(                           \
        initializer,                                                          \
        initializer.has_raw_data() ? initializer.raw_data().data() : nullptr, \
        initializer.has_raw_data() ? initializer.raw_data().size() : 0,       \
        reinterpret_cast<Y*>(unpackedTensor.get()), elementCount);            \
    if (!status.IsOK()) {                                                     \
      LOGS_DEFAULT(ERROR) << "Unpack tensor data failed.";                    \
    }                                                                         \
    break;                                                                    \
  }
  switch (initializer.data_type()) {
    CASE_PROTO(FLOAT, float);
    CASE_PROTO(DOUBLE, double);
    CASE_PROTO(BOOL, bool);
    CASE_PROTO(INT8, int8_t);
    CASE_PROTO(INT16, int16_t);
    CASE_PROTO(INT32, int32_t);
    CASE_PROTO(INT64, int64_t);
    CASE_PROTO(UINT8, uint8_t);
    CASE_PROTO(UINT16, uint16_t);
    CASE_PROTO(UINT32, uint32_t);
    CASE_PROTO(FLOAT16, onnxruntime::MLFloat16);
    default:
      return nullptr;
  }

  return unpackedTensor;
}

tim::vx::PadType GetPadType(const std::string type) {
  static const std::map<std::string, tim::vx::PadType> type_table = {
      {"NOTSET", tim::vx::PadType::AUTO},
      {"SAME_UPPER", tim::vx::PadType::SAME},
      {"SAME_LOWER", tim::vx::PadType::SAME},
      {"VALID", tim::vx::PadType::VALID},
  };
  auto search = type_table.find(type);
  if (search != type_table.end()) {
    return search->second;
  }
  return tim::vx::PadType::NONE;
}

bool ExcludeType(const NodeArg* node_arg, std::string& reason) {
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
    return false;
  }

  switch (type_proto->tensor_type().elem_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
      reason += "## only support int64 tensor as attribute.";
      return false;
    default:
      return true;
  }
}

bool CheckMainInputType(const Node* node, std::string& reason) {
  auto input_defs = node->InputDefs();
  return ExcludeType(input_defs[0], reason);
}

bool CheckZeroDim(const NodeArg* node_arg) {
  auto shape = node_arg->Shape();
  if (shape == nullptr || shape->dim_size() == 0) {
    return false;
  }
  for (int i = 0; i < shape->dim_size(); i++) {
    if (shape->dim(i).dim_value() == 0) {
      return false;
    }
  }
  return true;
}

bool CheckAllExcludeType(const Node* node, std::string& reason) {
  bool are_types_supported = true;
  node->ForEachDef(
      [&are_types_supported, &reason](const onnxruntime::NodeArg& node_arg,
                                      bool /*is_input*/) {
        are_types_supported &= ExcludeType(&node_arg, reason);
      });
  return are_types_supported;
}

bool CheckNoZeroDim(const Node* node) {
  bool no_zero_dim = true;

  node->ForEachDef(
      [&no_zero_dim](const onnxruntime::NodeArg& node_arg, bool /*is_input*/) {
        no_zero_dim &= vsi::npu::util::CheckZeroDim(&node_arg);
      });

  if (!no_zero_dim) {
    LOGS_DEFAULT(ERROR) <<"Tensor with dimension 0 is not supported.";
    return false;
  }
  return true;
}

}  // namespace util
}  // namespace npu
}  // namespace vsi
}  // namespace onnxruntime
