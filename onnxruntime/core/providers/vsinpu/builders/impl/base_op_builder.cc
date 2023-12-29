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
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
static bool IsTypeSupported(const NodeArg* node_arg) {
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
    return false;
  }

  switch (type_proto->tensor_type().elem_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
      return true;
    default:
      return false;
  }
}

bool BaseOpBuilder::IsSupported(const onnxruntime::GraphViewer& graph_viewer,
                                const Node* node) const {
  bool datatype_supported = true;
  node->ForEachDef([&datatype_supported](const onnxruntime::NodeArg& node_arg,
                                         bool /*is_input*/) {
    datatype_supported &= IsTypeSupported(&node_arg);
  });
  if (!datatype_supported) {
    LOGS_DEFAULT(VERBOSE) << "DataType is not supported by TIM-VX!";
    return false;
  }

  if (node->Domain() != "") {
    LOGS_DEFAULT(VERBOSE) << "Only support node with default domain!";
    return false;
  }

  if (!util::CheckNoZeroDim(node)) {
    return false;
  }

  return IsOpSupported(graph_viewer, node);
}

bool BaseOpBuilder::BuildOp(vsi::npu::GraphEP* graph_ep,
                            const onnxruntime::GraphViewer& graph_viewer,
                            const Node* node) {
  std::vector<std::shared_ptr<tim::vx::Tensor>> inputs;
  auto input_defs = node->InputDefs();
  for (auto input_def : input_defs) {
    auto it = std::find_if(
        graph_ep->GetGraphInputs().begin(), graph_ep->GetGraphInputs().end(),
        [input_def](const std::shared_ptr<GraphIOInfo>& info) {
          return info->name == input_def->Name();
        });
    tim::vx::TensorAttribute attr;
    if (graph_viewer.IsConstantInitializer(input_def->Name(), true)) {
      attr = tim::vx::TensorAttribute::CONSTANT;
    } else if (it == graph_ep->GetGraphInputs().end()) {
      attr = tim::vx::TensorAttribute::TRANSIENT;
    } else {
      attr = tim::vx::TensorAttribute::INPUT;
    }

    auto tensor = graph_ep->MapTIMVXTensor(graph_ep->GetGraph(), input_def,
                                           &graph_viewer, attr);
    inputs.push_back(tensor);
  }

  std::vector<std::shared_ptr<tim::vx::Tensor>> outputs;
  auto output_defs = node->OutputDefs();
  for (auto output_def : output_defs) {
    auto it = std::find_if(
        graph_ep->GetGraphOutputs().begin(), graph_ep->GetGraphOutputs().end(),
        [output_def](const std::shared_ptr<GraphIOInfo>& info) {
          return info->name == output_def->Name();
        });
    tim::vx::TensorAttribute attribute =
        it == graph_ep->GetGraphOutputs().end()
            ? tim::vx::TensorAttribute::TRANSIENT
            : tim::vx::TensorAttribute::OUTPUT;
    auto tensor = graph_ep->MapTIMVXTensor(graph_ep->GetGraph(), output_def,
                                           &graph_viewer, attribute);
    outputs.push_back(tensor);
  }
  return HandleBuildOp(graph_ep, inputs, outputs, node);
}
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
