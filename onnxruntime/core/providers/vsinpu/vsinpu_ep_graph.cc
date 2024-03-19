
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
#include "vsinpu_ep_graph.h"
#include "builders/op_builder_factory.h"
#include "vsinpu_util.h"

namespace onnxruntime {

namespace vsi {
namespace npu {
GraphEP::GraphEP() {
  context_ = tim::vx::Context::Create();
  graph_ = context_->CreateGraph();
  compiled_ = false;
}

bool GraphEP::SupportedOp(const onnxruntime::GraphViewer& graph_viewer,
                          const Node* node) {
  const auto& supported_builtins = vsi::npu::SupportedBuiltinOps();
  const auto& it = supported_builtins.find(node->OpType());
  if (supported_builtins.end() != it) {
    return it->second->IsSupported(graph_viewer, node);
  }

  LOGS_DEFAULT(WARNING) << "Fallback unsupported op " << node->OpType()
                        << "  to cpu.";
  return false;
}

void GraphEP::UpdateTensorMap(std::string name, std::shared_ptr<tim::vx::Tensor> dst_tensor) {
  auto it = tensors_.find(name);
  if (it != tensors_.end()) {
    it->second = dst_tensor;
  }
  for (auto& IO : graph_inputs_) {
    if (IO->name == name) {
      IO->tensor = dst_tensor;
      break;
    }
  }
  for (auto& IO : graph_outputs_) {
    if (IO->name == name) {
      IO->tensor = dst_tensor;
      break;
    }
  }
}

std::shared_ptr<NodeIOInfo> GraphEP::ConstructNodeIO(const std::shared_ptr<tim::vx::Operation>& op, std::vector<NodeArg*> input_arg, std::vector<NodeArg*> output_arg) {
  auto info = std::make_shared<vsi::npu::NodeIOInfo>();
  info->op_ = op;
  std::vector<std::string> input_names, output_names;
  if (input_arg.empty()) {
    info->input_names_ = std::vector<std::string>();
  } else {
    input_names.reserve(input_arg.size());
    std::transform(input_arg.begin(), input_arg.end(), std::back_inserter(input_names),
                   [](const NodeArg* node) -> std::string {
                     return node->Name();
                   });
    info->input_names_ = input_names;
  }
  if (output_arg.empty()) {
    info->output_names_ = std::vector<std::string>();
  } else {
    output_names.reserve(output_arg.size());
    std::transform(output_arg.begin(), output_arg.end(), std::back_inserter(output_names),
                   [](const NodeArg* node) -> std::string {
                     return node->Name();
                   });
    info->output_names_ = output_names;
  }

  return info;
}

bool GraphEP::BindTensors(const std::shared_ptr<NodeIOInfo>& nodeio_info) {
  auto op = nodeio_info->op_;
  auto input_names = nodeio_info->input_names_;
  auto output_names = nodeio_info->output_names_;
  if (!input_names.empty()) {
    for (auto name : input_names) {
      if (tensors_.find(name) == tensors_.end() || tensors_[name] == nullptr) {
        LOGS_DEFAULT(ERROR) << "Input tensor not defined or not found!";
        return false;
      }
      (*op).BindInput(tensors_[name]);
    }
  }
  if (!output_names.empty()) {
    for (auto name : output_names) {
      if (tensors_.find(name) == tensors_.end() || tensors_[name] == nullptr) {
        LOGS_DEFAULT(ERROR) << "Output tensor not defined or not found!";
        return false;
      }
      (*op).BindOutput(tensors_[name]);
    }
  }
  return true;
}

std::shared_ptr<tim::vx::Tensor> GraphEP::MapTIMVXTensor(
    std::shared_ptr<tim::vx::Graph>& graph, const NodeArg* arg,
    const GraphViewer* graph_viewer, tim::vx::TensorAttribute attribute) {
  if (tensors_.end() != tensors_.find(arg->Name())) {
    return tensors_.find(arg->Name())->second;
  } else {
    auto shape =
        vsi::npu::util::OnnxShapeToTIMVXShape(vsi::npu::util::GetTensorShape(*arg));
    std::reverse(shape.begin(), shape.end());
    tim::vx::DataType dt = vsi::npu::util::OnnxDtypeToTIMVXDtype(arg->Type());
    tim::vx::TensorSpec spec = tim::vx::TensorSpec(dt, shape, attribute);

    // quantize params setting
    std::shared_ptr<tim::vx::Tensor> tensor;
    if (attribute ==
        tim::vx::TensorAttribute::CONSTANT) {  // create const tensor
      const ONNX_NAMESPACE::TensorProto* tensor_proto =
          graph_viewer->GetConstantInitializer(arg->Name(), true);
      std::shared_ptr<uint8_t> unpackedTensor =
          vsi::npu::util::UnpackTensor(arg, *tensor_proto);

      const void* valueAddr =
          reinterpret_cast<const void*>(unpackedTensor.get());
      tensor = graph->CreateTensor(spec, valueAddr);

    } else
      tensor = graph->CreateTensor(spec);
    for (auto input : graph_inputs_) {
      if (input->name == arg->Name()) {
        input->tensor = tensor;
        input->shape = vsi::npu::util::GetTensorShape(*arg);
        break;
      }
    }
    for (auto output : graph_outputs_) {
      if (output->name == arg->Name()) {
        output->tensor = tensor;
        output->shape = utils::GetTensorShapeFromTensorShapeProto(*arg->Shape());
        break;
      }
    }
    tensors_.insert({arg->Name(), tensor});
    return tensor;
  }
}
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
