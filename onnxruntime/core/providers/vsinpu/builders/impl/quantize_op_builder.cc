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
#include "core/providers/vsinpu/builders/impl/quantize_op_builder.h"
#include <algorithm>

namespace onnxruntime {
namespace vsi {
namespace npu {
enum {
  input_tensor = 0,
  scale_tensor = 1,
  zero_point_tensor = 2
};

bool QuantizeLinearOpBuilder::IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                                            const Node* node) const {
  auto input_defs = node->InputDefs();
  auto scale_shape = vsi::npu::util::GetTensorShape(*input_defs[scale_tensor]);
  NodeAttrHelper helper(*node);
  if (helper.HasAttr("block_size") && helper.Get("block_size", 0) != 0) {
    LOGS_DEFAULT(WARNING) << "Not support block quantization.";
    return false;
  }
  if (!graph_viewer.IsInitializedTensor(input_defs[scale_tensor]->Name()) || (input_defs.size() == 3 && !graph_viewer.IsInitializedTensor(input_defs[zero_point_tensor]->Name()))) {
    LOGS_DEFAULT(WARNING) << "Only support const scale / zero point.";
    return false;
  }

  if (scale_shape.Size() != 1) {
    LOGS_DEFAULT(ERROR) << "Per channel quantized output is not supported in QuantizeLinearOp.";
    return false;
  }
  return true;
}

template <typename T1, typename T2>
struct QuantizeLinearOpBuilder::QuantizeImpl {
  QuantizeImpl(vsi::npu::GraphEP* graph_ep, std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
               std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs, const Node* node) {
    T1 scale;
    inputs[scale_tensor]->CopyDataFromTensor(&scale);
    T2 zero_point = 0;
    if (inputs.size() == 3) {
      inputs[zero_point_tensor]->CopyDataFromTensor(&zero_point);
    }
    tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, static_cast<float>(scale), static_cast<int32_t>(zero_point));
    tim::vx::TensorSpec OutSpec(outputs[0]->GetSpec());
    OutSpec.SetQuantization(quant);
    auto real_output = graph_ep->GetGraph()->CreateTensor(OutSpec);
    auto output_defs = node->OutputDefs();
    graph_ep->UpdateTensorMap(output_defs[0]->Name(), real_output);

    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::DataConvert>();
    std::vector<NodeArg*> input_defs;
    input_defs.push_back(util::RemoveWrapper(node->InputDefs()[input_tensor]));
    auto node_info = graph_ep->ConstructNodeIO(std::move(op), input_defs, util::RemoveWrapper(node->OutputDefs()));
    graph_ep->GetOps().push_back(node_info);
  }
};

bool QuantizeLinearOpBuilder::HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                                            std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                                            std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                                            const Node* node) {
  LOGS_DEFAULT(INFO) << "Creating Quantize Op.";
  NodeAttrHelper helper(*node);
  switch (inputs[scale_tensor]->GetDataType()) {
    case tim::vx::DataType::FLOAT32:
      switch (outputs[0]->GetDataType()) {
        case tim::vx::DataType::INT8:
          QuantizeImpl<float, int8_t>(graph_ep, inputs, outputs, node);
          break;
        case tim::vx::DataType::UINT8:
          QuantizeImpl<float, uint8_t>(graph_ep, inputs, outputs, node);
          break;
        case tim::vx::DataType::INT16:
          QuantizeImpl<float, int16_t>(graph_ep, inputs, outputs, node);
          break;
        case tim::vx::DataType::UINT16:
          QuantizeImpl<float, uint16_t>(graph_ep, inputs, outputs, node);
          break;
      }
      break;
    case tim::vx::DataType::FLOAT16:
      switch (outputs[0]->GetDataType()) {
        case tim::vx::DataType::INT8:
          QuantizeImpl<Ort::Float16_t, int8_t>(graph_ep, inputs, outputs, node);
          break;
        case tim::vx::DataType::UINT8:
          QuantizeImpl<Ort::Float16_t, uint8_t>(graph_ep, inputs, outputs, node);
          break;
        case tim::vx::DataType::INT16:
          QuantizeImpl<Ort::Float16_t, int16_t>(graph_ep, inputs, outputs, node);
          break;
        case tim::vx::DataType::UINT16:
          QuantizeImpl<Ort::Float16_t, uint16_t>(graph_ep, inputs, outputs, node);
          break;
      }
      break;
    case tim::vx::DataType::INT32:
      switch (outputs[0]->GetDataType()) {
        case tim::vx::DataType::INT8:
          QuantizeImpl<int32_t, int8_t>(graph_ep, inputs, outputs, node);
          break;
        case tim::vx::DataType::UINT8:
          QuantizeImpl<int32_t, uint8_t>(graph_ep, inputs, outputs, node);
          break;
        case tim::vx::DataType::INT16:
          QuantizeImpl<int32_t, int16_t>(graph_ep, inputs, outputs, node);
          break;
        case tim::vx::DataType::UINT16:
          QuantizeImpl<int32_t, uint16_t>(graph_ep, inputs, outputs, node);
          break;
      }
      break;
  }

  return true;
}

}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
