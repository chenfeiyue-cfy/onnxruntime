
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
#define ELEMENTWISE_OP_BUILDER(onnx_op_type, vsinpu_op_kind)                   \
  class onnx_op_type##OpBuilder : public BaseOpBuilder {                       \
    bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,                            \
                       std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,  \
                       std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs, \
                       const Node* node) override {                            \
      LOGS_DEFAULT(INFO) << "Creating " << #onnx_op_type << " Op";             \
      auto op = graph_ep->GetGraph()                                           \
                    ->CreateOperation<tim::vx::ops::vsinpu_op_kind>();         \
      (*op).BindInputs(inputs).BindOutputs(outputs);                           \
      graph_ep->GetOps().push_back(std::move(op));                             \
      return true;                                                             \
      ;                                                                        \
    }                                                                          \
  };
ELEMENTWISE_OP_BUILDER(Add, Add);
ELEMENTWISE_OP_BUILDER(Sub, Sub);
ELEMENTWISE_OP_BUILDER(Mul, Multiply);
ELEMENTWISE_OP_BUILDER(Div, Div);  // not consider zero
ELEMENTWISE_OP_BUILDER(Abs, Abs);
ELEMENTWISE_OP_BUILDER(Pow, Pow);
ELEMENTWISE_OP_BUILDER(Sqrt, Sqrt);
ELEMENTWISE_OP_BUILDER(Exp, Exp);
ELEMENTWISE_OP_BUILDER(Floor, Floor);
ELEMENTWISE_OP_BUILDER(Log, Log);
ELEMENTWISE_OP_BUILDER(Sin, Sin);
ELEMENTWISE_OP_BUILDER(HardSwish, HardSwish);
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
