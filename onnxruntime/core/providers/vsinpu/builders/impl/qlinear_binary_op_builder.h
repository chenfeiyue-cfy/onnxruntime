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
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
enum {
  INPUT_A = 0,
  INPUT_A_SCALE = 1,
  INPUT_A_ZP = 2,
  INPUT_B = 3,
  INPUT_B_SCALE = 4,
  INPUT_B_ZP = 5,
  OUTPUT_SCALE = 6,
  OUTPUT_ZP = 7,
};
template <typename T>
struct QBinaryImpl {
  QBinaryImpl(vsi::npu::GraphEP* graph_ep, std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
              std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs, const Node* node, const std::shared_ptr<tim::vx::Operation>& op) {
    float A_scale, B_scale, out_scale;

    inputs[INPUT_A_SCALE]->CopyDataFromTensor(&A_scale);
    inputs[INPUT_B_SCALE]->CopyDataFromTensor(&B_scale);
    inputs[OUTPUT_SCALE]->CopyDataFromTensor(&out_scale);
    T A_zp, B_zp, out_zp;
    inputs[INPUT_A_ZP]->CopyDataFromTensor(&A_zp);
    inputs[INPUT_B_ZP]->CopyDataFromTensor(&B_zp);
    inputs[OUTPUT_ZP]->CopyDataFromTensor(&out_zp);
    tim::vx::Quantization AQuant(tim::vx::QuantType::ASYMMETRIC, A_scale, static_cast<int32_t>(A_zp));
    tim::vx::Quantization BQuant(tim::vx::QuantType::ASYMMETRIC, B_scale, static_cast<int32_t>(B_zp));
    tim::vx::Quantization OutQuant(tim::vx::QuantType::ASYMMETRIC, out_scale, static_cast<int32_t>(out_zp));
    tim::vx::TensorSpec ASpec(inputs[INPUT_A]->GetSpec());
    tim::vx::TensorSpec BSpec(inputs[INPUT_B]->GetSpec());
    tim::vx::TensorSpec OutputSpec(outputs[0]->GetSpec());
    ASpec.SetQuantization(AQuant);
    BSpec.SetQuantization(BQuant);
    OutputSpec.SetQuantization(OutQuant);
    auto A_tensor = graph_ep->GetGraph()->CreateTensor(ASpec);
    auto B_tensor = graph_ep->GetGraph()->CreateTensor(BSpec);
    auto output_tensor = graph_ep->GetGraph()->CreateTensor(OutputSpec);
    if (inputs[INPUT_B]->IsConstTensor()) {
      std::vector<T> B_data(inputs[INPUT_B]->GetSpec().GetElementNum());
      inputs[INPUT_B]->CopyDataFromTensor(B_data.data());
      B_tensor->CopyDataToTensor(B_data.data());
    }
    graph_ep->UpdateTensorMap(node->InputDefs()[INPUT_A]->Name(), A_tensor);
    graph_ep->UpdateTensorMap(node->InputDefs()[INPUT_B]->Name(), B_tensor);
    graph_ep->UpdateTensorMap(node->OutputDefs()[0]->Name(), output_tensor);
    std::vector<NodeArg*> input_defs;
    input_defs.push_back(util::RemoveWrapper(node->InputDefs()[INPUT_A]));
    input_defs.push_back(util::RemoveWrapper(node->InputDefs()[INPUT_B]));
    auto node_info = graph_ep->ConstructNodeIO(std::move(op), input_defs, util::RemoveWrapper(node->OutputDefs()));
    graph_ep->GetOps().push_back(node_info);
  }
};

class BaseQLinearOpBuilder : public BaseOpBuilder {
 protected:
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer, const Node* node) const override {
    for (int i = 0; i < node->InputDefs().size(); i++) {
      if (i == INPUT_A || i == INPUT_B) continue;
      if (!graph_viewer.IsConstantInitializer(node->InputDefs()[i]->Name(), true)) {
        LOGS_DEFAULT(WARNING) << "Only support const scale / zero point.";
        return false;
      }
    }
    return true;
  }
};

class QLinearAddOpBuilder : public BaseQLinearOpBuilder {
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const Node* node) override {
    LOGS_DEFAULT(VERBOSE) << "Creating QLinearAdd Op.";
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Add>();
    switch (inputs[INPUT_A]->GetDataType()) {
      case tim::vx::DataType::INT8:
        QBinaryImpl<int8_t>(graph_ep, inputs, outputs, node, op);
        break;
      case tim::vx::DataType::UINT8:
        QBinaryImpl<uint8_t>(graph_ep, inputs, outputs, node, op);
        break;
    }
    return true;
  }
};

class QLinearMulOpBuilder : public BaseQLinearOpBuilder {
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const Node* node) override {
    LOGS_DEFAULT(VERBOSE) << "Creating QLinearMul Op.";
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Multiply>();
    switch (inputs[INPUT_A]->GetDataType()) {
      case tim::vx::DataType::INT8:
        QBinaryImpl<int8_t>(graph_ep, inputs, outputs, node, op);
        break;
      case tim::vx::DataType::UINT8:
        QBinaryImpl<uint8_t>(graph_ep, inputs, outputs, node, op);
        break;
    }
    return true;
  }
};

}  // namespace npu
}  // namespace vsi
}  // namespace onnxruntime
