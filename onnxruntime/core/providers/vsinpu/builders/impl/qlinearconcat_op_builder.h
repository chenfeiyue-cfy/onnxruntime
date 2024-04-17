/****************************************************************************
 *
 *    Copyright (c) 2024 Vivante Corporation
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
#include "core/providers/shared/utils/utils.h"
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"
#define start_id 2
namespace onnxruntime {
namespace vsi {
namespace npu {

enum InputIndex {
  INPUT_DATA = 0,
  INPUT_SCALE = 1,
  INPUT_ZP = 2
};

template <typename T>
std::shared_ptr<tim::vx::Tensor> CreateAndAddInputTensor(vsi::npu::GraphEP* graph_ep, std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs, const std::string& name) {
  float scale;
  T zp;

  inputs[INPUT_SCALE]->CopyDataFromTensor(&scale);
  inputs[INPUT_ZP]->CopyDataFromTensor(&zp);

  tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, scale, static_cast<int32_t>(zp));
  tim::vx::TensorSpec spec(inputs[INPUT_DATA]->GetDataType(), inputs[INPUT_DATA]->GetShape(), inputs[INPUT_DATA]->GetSpec().GetTensorAttribute(), quant);

  auto tensor = graph_ep->GetGraph()->CreateTensor(spec);
  if (inputs[INPUT_DATA]->IsConstTensor()) {
    std::vector<T> data(inputs[INPUT_DATA]->GetSpec().GetElementNum());
    inputs[INPUT_DATA]->CopyDataFromTensor(data.data());
    tensor->CopyDataToTensor(data.data());
  }

  graph_ep->UpdateTensorMap(name, tensor);
  return tensor;
}

template <typename T>
void ProcessInputsAndCreateOperation(vsi::npu::GraphEP* graph_ep, std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs, std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs, const Node* node) {
  std::vector<NodeArg*> input_defs;
  for (size_t i = start_id; i < inputs.size(); i += 3) {
    auto name = node->InputDefs()[i]->Name();
    std::vector<std::shared_ptr<tim::vx::Tensor>> sub_inputs{inputs[i], inputs[i + 1], inputs[i + 2]};
    auto real_input = CreateAndAddInputTensor<T>(graph_ep, sub_inputs, name);
    input_defs.push_back(util::RemoveWrapper(node->InputDefs()[i]));
  }

  float out_scale;
  T out_zp;

  inputs[0]->CopyDataFromTensor(&out_scale);
  inputs[1]->CopyDataFromTensor(&out_zp);

  auto OutSpec = outputs[0]->GetSpec();
  tim::vx::Quantization OutQuant(tim::vx::QuantType::ASYMMETRIC, out_scale, static_cast<int32_t>(out_zp));
  OutSpec.SetQuantization(OutQuant);

  auto real_output = graph_ep->GetGraph()->CreateTensor(OutSpec);
  graph_ep->UpdateTensorMap(node->OutputDefs()[0]->Name(), real_output);

  NodeAttrHelper helper(*node);
  int axis = helper.Get("axis", 0);
  axis = util::ReverseAxis(axis, inputs[start_id]->GetShape().size());
  auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Concat>(axis, (inputs.size() - 2) / 3);
  auto node_info = graph_ep->ConstructNodeIO(std::move(op), input_defs, util::RemoveWrapper(node->OutputDefs()));

  graph_ep->GetOps().push_back(node_info);
}

class QLinearConcatOpBuilder : public BaseOpBuilder {
 protected:
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer, const Node* node) const override {
    if (!graph_viewer.IsConstantInitializer(node->InputDefs()[0]->Name(), true)){
      LOGS_DEFAULT(WARNING) << "Output scale must be known.";
      return false;
    }
    if (!graph_viewer.IsConstantInitializer(node->InputDefs()[1]->Name(), true)){
      LOGS_DEFAULT(WARNING) << "Output zp must be known.";
      return false;
    }
    return true;
  }

  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep, std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs, std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs, const Node* node) override {
    LOGS_DEFAULT(VERBOSE) << "Creating QLinearConcat Op.";
    switch (inputs[start_id]->GetDataType()) {
      case tim::vx::DataType::INT8:
        ProcessInputsAndCreateOperation<int8_t>(graph_ep, inputs, outputs, node);
        break;
      case tim::vx::DataType::UINT8:
        ProcessInputsAndCreateOperation<uint8_t>(graph_ep, inputs, outputs, node);
        break;
    }
    return true;
  }
};

}  // namespace npu
}  // namespace vsi
}  // namespace onnxruntime
