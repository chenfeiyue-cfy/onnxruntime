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
class ConvOpBuilder : public BaseOpBuilder {
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    auto input_defs = node->InputDefs();
    auto shape = vsi::npu::util::GetTensorShape(*input_defs[0]);
    if (shape.NumDimensions() == 5) {
      LOGS_DEFAULT(ERROR) << "Not support conv3d yet.";
      return false;
    }
    return true;
  }

  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const Node* node) override {
    auto input_tensor = inputs[0];
    auto weight_tensor = inputs[1];
    const bool is_1d_conv =
        weight_tensor->GetShape().size() == 3 ? true : false;
    NodeAttrHelper helper(*node);
    auto padtype = helper.Get("auto_pad", std::string(""));
    auto group = helper.Get("group", static_cast<uint32_t>(1));

    auto op_type = group != 1 ? (is_1d_conv ? "GroupConv1D" : "GroupConv2D")
                              : (is_1d_conv ? "Conv1D" : "Conv2D");
    std::string op_name = std::string("Creating ") + op_type + " Op";
    LOGS_DEFAULT(INFO) << op_name;

    uint32_t default_uint = 1;
    std::vector<uint32_t> default_vec = {1, 1};

    auto stride =
        helper.Get("strides", is_1d_conv ? std::vector<uint32_t>{default_uint}
                                         : default_vec);
    auto dilation =
        helper.Get("dilations", is_1d_conv ? std::vector<uint32_t>{default_uint}
                                           : default_vec);

    std::shared_ptr<tim::vx::Operation> op;
    if (padtype != "NOTSET") {  // array "pads" is not set
      if (group != 1) {
        if (is_1d_conv) {
          op = graph_ep->GetGraph()
                   ->CreateOperation<tim::vx::ops::GroupedConv1d>(
                       vsi::npu::util::GetPadType(padtype), stride[0],
                       dilation[0], group, tim::vx::DataLayout::WCN,
                       tim::vx::DataLayout::WIcOc);
        } else {
          op = graph_ep->GetGraph()
                   ->CreateOperation<tim::vx::ops::GroupedConv2d>(
                       vsi::npu::util::GetPadType(padtype),
                       /* W_stride, H_stride*/
                       std::array<uint32_t, 2>{stride[1], stride[0]},
                       /* W_dilation, H_dilation*/
                       std::array<uint32_t, 2>{dilation[1], dilation[0]}, group,
                       tim::vx::DataLayout::WHCN, tim::vx::DataLayout::WHIcOc);
        }
      } else {
        if (is_1d_conv) {
          op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Conv1d>(
              vsi::npu::util::GetPadType(padtype), stride[0], dilation[0], 0,
              tim::vx::DataLayout::WCN, tim::vx::DataLayout::WIcOc);
        } else {
          op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Conv2d>(
              vsi::npu::util::GetPadType(padtype),
              /* W_stride, H_stride*/
              std::array<uint32_t, 2>{stride[1], stride[0]},
              /* W_dilation, H_dilation*/
              std::array<uint32_t, 2>{dilation[1], dilation[0]}, 0,
              tim::vx::DataLayout::WHCN, tim::vx::DataLayout::WHIcOc);
        }
      }
    } else {
      auto pads = helper.Get("pads", std::vector<uint32_t>{0U, 0U});
      if (group != 1) {
        if (is_1d_conv) {
          op = graph_ep->GetGraph()
                   ->CreateOperation<tim::vx::ops::GroupedConv1d>(
                       vsi::npu::util::GetPadType(padtype),
                       std::array<uint32_t, 2>{pads[0], pads[1]}, stride[0],
                       dilation[0], group, tim::vx::DataLayout::WCN,
                       tim::vx::DataLayout::WIcOc);
        } else {
          op = graph_ep->GetGraph()
                   ->CreateOperation<tim::vx::ops::GroupedConv2d>(
                       /* W_begin,W_end, H_begin,H_end*/ std::array<
                           uint32_t, 4>{pads[1], pads[3], pads[0], pads[2]},
                       /* W_stride, H_stide*/
                       std::array<uint32_t, 2>{stride[1], stride[0]},
                       /* W_dilation, H_dilation*/
                       std::array<uint32_t, 2>{dilation[1], dilation[0]}, group,
                       tim::vx::DataLayout::WHCN, tim::vx::DataLayout::WHIcOc);
        }
      } else {
        if (is_1d_conv) {
          op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Conv1d>(
              std::array<uint32_t, 2>{pads[0], pads[1]}, stride[0], dilation[0],
              0, tim::vx::DataLayout::WCN, tim::vx::DataLayout::WIcOc);
        } else {
          op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Conv2d>(
              /* W_begin,W_end, H_begin,H_end*/ std::array<uint32_t,
                                                           4>{pads[1], pads[3],
                                                              pads[0], pads[2]},
              /* W_stride, H_stride*/
              std::array<uint32_t, 2>{stride[1], stride[0]},
              /* W_dilation, H_dilation*/
              std::array<uint32_t, 2>{dilation[1], dilation[0]}, 0,
              tim::vx::DataLayout::WHCN, tim::vx::DataLayout::WHIcOc);
        }
      }
    }
    auto node_info = graph_ep->ConstructNodeIO(std::move(op), util::RemoveWrapper(node->InputDefs()), util::RemoveWrapper(node->OutputDefs()));
    graph_ep->GetOps().push_back(node_info);
    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
