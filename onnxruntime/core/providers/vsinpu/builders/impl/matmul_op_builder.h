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
class MatMulOpBuilder : public BaseOpBuilder {
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    auto output_defs = node->OutputDefs();
    if (output_defs[0]->Shape()->dim_size() == 0) {
      LOGS_DEFAULT(ERROR) << "Inner product of 1-D tensor is not supported in MatMul op.";
      return false;
    }
    for(auto input : node ->InputDefs()){
      if(*input->Type() == "tensor(int64)") {
        LOGS_DEFAULT(WARNING) << "Cannot support int64 Gemm operation.";
        return false;
      }
    }
    return true;
  }
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const Node* node) override {
    LOGS_DEFAULT(VERBOSE) << "Creating Matmul Op.";
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Matmul>();
    auto node_info = graph_ep->ConstructNodeIO(std::move(op), util::RemoveWrapper(node->InputDefs()), util::RemoveWrapper(node->OutputDefs()));
    graph_ep->GetOps().push_back(node_info);
    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
