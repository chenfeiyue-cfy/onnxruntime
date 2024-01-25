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
#include "core/framework/tensorprotoutils.h"
#include <variant>
namespace onnxruntime {
namespace vsi {
namespace npu {
class QLinearConvOpBuilder : public BaseOpBuilder {
  enum {
    INPUT_TENSOR = 0,
    INPUT_TENSOR_SCALE = 1,
    INPUT_TENSOR_ZP = 2,
    WEIGHT_TENSOR = 3,
    WEIGHT_TENSOR_SCALE = 4,
    WEIGHT_TENSOR_ZP = 5,
    OUTPUT_TENSOR_SCALE = 6,
    OUTPUT_TENSOR_ZP = 7,
    BIAS_TENSOR = 8,
  };

  template <typename T>
  std::vector<T> getParamAsVector(std::shared_ptr<tim::vx::Tensor> qt_params) {
    std::vector<T> values(qt_params->GetSpec().GetElementNum());
    qt_params->CopyDataFromTensor(values.data());
    return values;
  }

  template <typename T>
  T getParamAsScalar(std::shared_ptr<tim::vx::Tensor> qt_params) {
    T val;
    qt_params->CopyDataFromTensor(&val);
    return val;
  }

  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    auto input_defs = node->InputDefs();
    auto input_shape = vsi::npu::util::GetTensorShape(*input_defs[INPUT_TENSOR]);
    auto w_scale_shape = vsi::npu::util::GetTensorShape(*input_defs[WEIGHT_TENSOR_SCALE]);
    if (input_shape.NumDimensions() != 4) {
      LOGS_DEFAULT(ERROR) << "Not support conv3d&& conv1d yet.";
      return false;
    }

    if (!graph_viewer.IsInitializedTensor(input_defs[INPUT_TENSOR_SCALE]->Name()) || !graph_viewer.IsInitializedTensor(input_defs[WEIGHT_TENSOR]->Name())) {
      LOGS_DEFAULT(ERROR) << "Not support quantization definitions or weights that are not constant yet.";
      return false;
    }

    if (w_scale_shape.Size() != 1 && *input_defs[WEIGHT_TENSOR]->Type() == "tensor(int8)") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto =
          graph_viewer.GetConstantInitializer(input_defs[WEIGHT_TENSOR_ZP]->Name(), true);
      std::vector<int8_t> w_zp(1);
      auto status = onnxruntime::utils::UnpackTensor(
          *tensor_proto,
          tensor_proto->has_raw_data() ? tensor_proto->raw_data().data() : nullptr,
          tensor_proto->has_raw_data() ? tensor_proto->raw_data().size() : 0,
          w_zp.data(), w_zp.size());
      if (!status.IsOK()) {
        LOGS_DEFAULT(ERROR) << "Failed to get data from weight zp tensor.";
        return false;
      }
      if (w_zp[0] != 0) {
        LOGS_DEFAULT(ERROR) << "Asymmetric perchannel quantization with datatype int8 is not supported.";
        return false;
      }
    }
    return true;
  }
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const Node* node) override {
    LOGS_DEFAULT(VERBOSE) << "Creating QLinearConv Op.";
    auto x_scale = getParamAsScalar<float>(inputs[INPUT_TENSOR_SCALE]);
    auto y_scale = getParamAsScalar<float>(inputs[OUTPUT_TENSOR_SCALE]);
    std::variant<int8_t, uint8_t> x_zp, y_zp;
    if (inputs[WEIGHT_TENSOR]->GetDataType() == tim::vx::DataType::INT8) {
      x_zp = getParamAsScalar<int8_t>(inputs[INPUT_TENSOR_ZP]);
    } else
      x_zp = getParamAsScalar<uint8_t>(inputs[INPUT_TENSOR_ZP]);
    if (outputs[0]->GetDataType() == tim::vx::DataType::INT8) {
      y_zp = getParamAsScalar<int8_t>(inputs[OUTPUT_TENSOR_ZP]);
    } else
      y_zp = getParamAsScalar<uint8_t>(inputs[OUTPUT_TENSOR_ZP]);

    // quantization of W can be perchanneled , which means w_scale could be a 1-D tensor.
    bool is_pcq = inputs[WEIGHT_TENSOR_SCALE]->GetSpec().GetElementNum() == 1 ? false : true;
    tim::vx::Quantization WeightQuant;
    tim::vx::Quantization BiasQuant;
    std::vector<int32_t> biasdata(inputs.size() == 9 ? inputs[BIAS_TENSOR]->GetSpec().GetElementNum() : 1);
    if (is_pcq) {
      auto w_scale = getParamAsVector<float>(inputs[WEIGHT_TENSOR_SCALE]);
      std::variant<std::vector<int8_t>, std::vector<uint8_t>> w_zp;
      if (inputs[WEIGHT_TENSOR]->GetDataType() == tim::vx::DataType::INT8) {
        w_zp = getParamAsVector<int8_t>(inputs[WEIGHT_TENSOR_ZP]);
      } else
        w_zp = getParamAsVector<uint8_t>(inputs[WEIGHT_TENSOR_ZP]);
      int32_t value = std::visit([](auto& vec) {
        return static_cast<int32_t>(vec[0]);
      }, w_zp);
      std::vector<int32_t> timvx_w_zp(w_scale.size(), value);
      if (timvx_w_zp[0] != 0) {
        WeightQuant.SetType(tim::vx::QuantType::ASYMMETRIC_PER_CHANNEL);
        WeightQuant.SetChannelDim(3);
        WeightQuant.SetScales(w_scale);
        WeightQuant.SetZeroPoints(timvx_w_zp);
      } else {
        WeightQuant.SetType(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL);
        WeightQuant.SetChannelDim(3);
        WeightQuant.SetScales(w_scale);
        WeightQuant.SetZeroPoints(timvx_w_zp);
      }
      if (inputs.size() == 9) {
        for (auto& val : w_scale) {
          val = val * x_scale;
        }
        BiasQuant.SetType(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL);
        BiasQuant.SetChannelDim(0);
        BiasQuant.SetScales(w_scale);
        BiasQuant.SetZeroPoints({0});
      }
    } else {
      auto w_scale = getParamAsScalar<float>(inputs[WEIGHT_TENSOR_SCALE]);
      std::variant<int8_t, uint8_t> w_zp;
      if (inputs[WEIGHT_TENSOR]->GetDataType() == tim::vx::DataType::INT8) {
        w_zp = getParamAsScalar<int8_t>(inputs[WEIGHT_TENSOR_ZP]);
      } else
        w_zp = getParamAsScalar<uint8_t>(inputs[WEIGHT_TENSOR_ZP]);
      int32_t timvx_w_zp = std::visit([](auto arg) -> int32_t { return static_cast<int32_t>(arg); }, w_zp);
      WeightQuant.SetType(tim::vx::QuantType::ASYMMETRIC);
      WeightQuant.SetScales({w_scale});
      WeightQuant.SetZeroPoints({timvx_w_zp});
      if (inputs.size() == 9) {
        BiasQuant.SetType(tim::vx::QuantType::ASYMMETRIC);
        ;
        BiasQuant.SetScales({x_scale * w_scale});
        BiasQuant.SetZeroPoints({0});
      }
    }
    int32_t timvx_x_zp = std::visit([](auto arg) -> int32_t { return static_cast<int32_t>(arg); }, x_zp);
    int32_t timvx_y_zp = std::visit([](auto arg) -> int32_t { return static_cast<int32_t>(arg); }, y_zp);
    tim::vx::Quantization InputQuant(tim ::vx::QuantType::ASYMMETRIC, x_scale, timvx_x_zp);
    tim::vx::Quantization OutputQuant(tim ::vx::QuantType::ASYMMETRIC, y_scale, timvx_y_zp);
    tim::vx::TensorSpec InputSpec(inputs[INPUT_TENSOR]->GetSpec());
    InputSpec.SetQuantization(InputQuant);
    tim::vx::TensorSpec WeightSpec(inputs[WEIGHT_TENSOR]->GetSpec());
    WeightSpec.SetQuantization(WeightQuant);
    tim::vx::TensorSpec OutputSpec(outputs[0]->GetSpec());
    OutputSpec.SetQuantization(OutputQuant);
    auto input_tensor = graph_ep->GetGraph()->CreateTensor(InputSpec);
    auto weight_tensor = graph_ep->GetGraph()->CreateTensor(WeightSpec);
    auto output_tensor = graph_ep->GetGraph()->CreateTensor(OutputSpec);
    std::vector<uint8_t> weight_data(inputs[WEIGHT_TENSOR]->GetSpec().GetElementNum());
    inputs[WEIGHT_TENSOR]->CopyDataFromTensor(weight_data.data());
    weight_tensor->CopyDataToTensor(weight_data.data());

    NodeAttrHelper helper(*node);
    auto padtype = helper.Get("auto_pad", std::string(""));
    auto group = helper.Get("group", static_cast<uint32_t>(1));
    std::vector<uint32_t> default_vec = {1, 1, 1, 1};
    auto stride =
        helper.Get("strides", default_vec);
    auto dilation =
        helper.Get("dilations", default_vec);
    std::shared_ptr<tim::vx::Operation> op;
    if (padtype != "NOTSET") {  // array "pads" is not set
      if (group != 1 && group != weight_tensor->GetShape()[3]) {
        op = graph_ep->GetGraph()
                 ->CreateOperation<tim::vx::ops::GroupedConv2d>(
                     vsi::npu::util::GetPadType(padtype),
                     std::array<uint32_t, 2>{stride[1], stride[0]},
                     std::array<uint32_t, 2>{dilation[1], dilation[0]}, group,
                     tim::vx::DataLayout::WHCN, tim::vx::DataLayout::WHIcOc);

      } else {
        int32_t multiplier = group == 1 ? 0 : weight_tensor->GetShape()[3] / input_tensor->GetShape()[2];
        op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Conv2d>(
            vsi::npu::util::GetPadType(padtype),
            std::array<uint32_t, 2>{stride[1], stride[0]},
            std::array<uint32_t, 2>{dilation[1], dilation[0]}, multiplier,
            tim::vx::DataLayout::WHCN, tim::vx::DataLayout::WHIcOc);
      }
    } else {
      std::vector<uint32_t> default_pads(4, 0);
      auto pads = helper.Get("pads", default_pads);
      if (group != 1 && group != weight_tensor->GetShape()[3]) {
        op = graph_ep->GetGraph()
                 ->CreateOperation<tim::vx::ops::GroupedConv2d>(
                     std::array<uint32_t, 4>{pads[1], pads[3], pads[0], pads[2]},
                     std::array<uint32_t, 2>{stride[1], stride[0]},
                     std::array<uint32_t, 2>{dilation[1], dilation[0]}, group,
                     tim::vx::DataLayout::WHCN, tim::vx::DataLayout::WHIcOc);

      } else {
        int32_t multiplier = group == 1 ? 0 : weight_tensor->GetShape()[3] / input_tensor->GetShape()[2];
        op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Conv2d>(
            std::array<uint32_t, 4>{pads[1], pads[3],
                                    pads[0], pads[2]},
            std::array<uint32_t, 2>{stride[1], stride[0]},
            std::array<uint32_t, 2>{dilation[1], dilation[0]}, multiplier,
            tim::vx::DataLayout::WHCN, tim::vx::DataLayout::WHIcOc);
      }
    }

    if (inputs.size() == 9) {
      tim::vx::TensorSpec BiasSpec(inputs[BIAS_TENSOR]->GetSpec());
      BiasSpec.SetQuantization(BiasQuant);
      inputs[8]->CopyDataFromTensor(biasdata.data());
      auto bias_tensor = graph_ep->GetGraph()->CreateTensor(BiasSpec, biasdata.data());
      op->BindInput(input_tensor).BindInput(weight_tensor).BindInput(bias_tensor).BindOutput(output_tensor);
    } else {
      op->BindInput(input_tensor).BindInput(weight_tensor).BindOutput(output_tensor);
    }

    for (auto& IO : graph_ep->GetGraphInputs()) {
      if (IO->tensor.get() == inputs[0].get()) {
        IO->tensor = input_tensor;
      }
    }
    for (auto& IO : graph_ep->GetGraphOutputs()) {
      if (IO->tensor.get() == outputs[0].get()) {
        IO->tensor = output_tensor;
      }
    }
    outputs[0] = output_tensor;
    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
