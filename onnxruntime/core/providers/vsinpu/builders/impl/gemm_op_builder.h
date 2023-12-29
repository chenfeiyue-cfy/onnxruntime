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
#include "core/providers/shared/utils/utils.h"
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"
namespace onnxruntime {
namespace vsi {
namespace npu {
class GemmOpBuilder : public BaseOpBuilder {
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const Node* node) override {
    LOGS_DEFAULT(VERBOSE) << "Creating Gemm Op.";
    auto input_A = inputs[0];
    auto input_B = inputs[1];
    NodeAttrHelper helper(*node);
    float default_float = 1.0f;
    auto alpha = helper.Get("alpha", default_float);
    auto beta = helper.Get("beta", default_float);
    auto trans_A = helper.Get("transA", 0);
    auto trans_B = helper.Get("transB", 0);
    bool has_alpha = (alpha != 1.0);
    bool has_beta = (beta != 1.0);
    bool has_C = (inputs.size() == 3);

    tim::vx::TensorSpec CoefSpec(tim::vx::DataType::FLOAT32, {1},
                                 tim::vx::TensorAttribute::CONSTANT);
    auto alpha_tensor = graph_ep->GetGraph()->CreateTensor(CoefSpec);
    alpha_tensor->CopyDataToTensor(&alpha);

    auto beta_tensor = graph_ep->GetGraph()->CreateTensor(CoefSpec);
    beta_tensor->CopyDataToTensor(&beta);

    auto updatedA = input_A;
    auto updatedB = input_B;
    if (has_alpha) {
      updatedA = graph_ep->GetGraph()->CreateTensor(
          input_A->GetSpec().AsTransientSpec());
      auto mul1 =
          graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Multiply>();
      (*mul1).BindInput(input_A).BindInput(alpha_tensor).BindOutput(updatedA);
      graph_ep->GetOps().push_back(std::move(mul1));
    }
    if (has_beta) {
      updatedB = graph_ep->GetGraph()->CreateTensor(
          input_B->GetSpec().AsTransientSpec());
      auto mul2 =
          graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Multiply>();
      (*mul2).BindInput(input_B).BindInput(beta_tensor).BindOutput(updatedA);
      graph_ep->GetOps().push_back(std::move(mul2));
    }

    if (has_C) {
      auto AB_output = graph_ep->GetGraph()->CreateTensor(
          outputs[0]->GetSpec().AsTransientSpec());
      auto matmul = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Matmul>(
          trans_A, trans_B);
      (*matmul).BindInput(updatedA).BindInput(updatedB).BindOutput(AB_output);
      graph_ep->GetOps().push_back((std::move(matmul)));
      auto add = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Add>();
      (*add).BindInput(AB_output).BindInput(inputs[2]).BindOutput(outputs[0]);
      graph_ep->GetOps().push_back((std::move(add)));
    } else {
      auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Matmul>(
          trans_A, trans_B);
      (*op).BindInput(updatedA).BindInput(updatedB).BindOutput(outputs[0]);
      graph_ep->GetOps().push_back((std::move(op)));
    }
    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
