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
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    NodeAttrHelper helper(*node);
    auto trans_A = helper.Get("transA", 0);
    auto trans_B = helper.Get("transB", 0);
    if (trans_A == trans_B && trans_A == 1) {
      LOGS_DEFAULT(WARNING) << "Cannot support Gemm Op with transA && transB both be true.";
      return false;
    }
    return true;
  }
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

    auto matmul_impl = [&](std::shared_ptr<tim::vx::Tensor> input_A,
                           std::shared_ptr<tim::vx::Tensor> input_B,
                           std::shared_ptr<tim::vx::Tensor> output) {
      auto matmul_op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Matmul>(
          trans_A, trans_B);
      (*matmul_op).BindInput(input_A).BindInput(input_B).BindOutput(output);
      graph_ep->GetOps().push_back(std::move(matmul_op));
    };

    auto multiply_impl = [&](std::shared_ptr<tim::vx::Tensor> input,
                             std::shared_ptr<tim::vx::Tensor> coef,
                             std::shared_ptr<tim::vx::Tensor> output) {
      auto multiply_op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Multiply>();
      (*multiply_op).BindInput(input).BindInput(coef).BindOutput(output);
      graph_ep->GetOps().push_back(std::move(multiply_op));
    };

    auto add_impl = [&](std::shared_ptr<tim::vx::Tensor> input_A,
                        std::shared_ptr<tim::vx::Tensor> input_B,
                        std::shared_ptr<tim::vx::Tensor> output) {
      auto add_op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Add>();
      (*add_op).BindInput(input_A).BindInput(input_B).BindOutput(output);
      graph_ep->GetOps().push_back(std::move(add_op));
    };

    auto AB_output = outputs[0];
    if (has_alpha) {
      AB_output = graph_ep->GetGraph()->CreateTensor(
          outputs[0]->GetSpec().AsTransientSpec());
      matmul_impl(input_A, input_B, AB_output);

      if (has_C) {
        auto mul1_output = graph_ep->GetGraph()->CreateTensor(
            outputs[0]->GetSpec().AsTransientSpec());
        multiply_impl(AB_output, alpha_tensor, mul1_output);
        if (has_beta) {
          auto multiplied_C = graph_ep->GetGraph()->CreateTensor(
              inputs[2]->GetSpec().AsTransientSpec());
          multiply_impl(inputs[2], beta_tensor, multiplied_C);
          add_impl(mul1_output, multiplied_C, outputs[0]);
        } else {
          add_impl(mul1_output, inputs[2], outputs[0]);
        }
      } else {
        multiply_impl(AB_output, alpha_tensor, outputs[0]);
      }
    } else {
      if (has_C) {
        AB_output = graph_ep->GetGraph()->CreateTensor(
            outputs[0]->GetSpec().AsTransientSpec());
        matmul_impl(input_A, input_B, AB_output);
        if (has_beta) {
          auto multiplied_C = graph_ep->GetGraph()->CreateTensor(
              inputs[2]->GetSpec().AsTransientSpec());
          multiply_impl(inputs[2], beta_tensor, multiplied_C);
          add_impl(AB_output, multiplied_C, outputs[0]);
        } else {
          add_impl(AB_output, inputs[2], outputs[0]);
        }
      } else {
        matmul_impl(input_A, input_B, outputs[0]);
      }
    }

    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
