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
#include "core/providers/vsinpu/builders/impl/qlinearmatmul_op_builder.h"

namespace onnxruntime {
namespace vsi {
namespace npu {

template <typename T1, typename T2, typename T3>
struct QLinearMatMulOpBuilder::QMatMulImpl {
  QMatMulImpl(vsi::npu::GraphEP* graph_ep, std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
              std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs, const Node* node) {
    T1 A_zp;
    inputs[A_zero_point]->CopyDataFromTensor(&A_zp);
    T2 B_zp;
    inputs[B_zero_point]->CopyDataFromTensor(&B_zp);
    T3 out_zp;
    inputs[out_zero_point]->CopyDataFromTensor(&out_zp);
    tim::vx::Quantization AQuant(tim::vx::QuantType::ASYMMETRIC, static_cast<float>(1.0f), static_cast<int32_t>(A_zp));
    tim::vx::Quantization BQuant(tim::vx::QuantType::ASYMMETRIC, static_cast<float>(1.0f), static_cast<int32_t>(B_zp));
    tim::vx::Quantization OutQuant(tim::vx::QuantType::ASYMMETRIC, static_cast<float>(1.0f), static_cast<int32_t>(out_zp));

    switch (inputs[A_scale]->GetDataType()) {
      case tim::vx::DataType::FLOAT32: {
        float a_scale, b_scale, o_scale;
        inputs[A_scale]->CopyDataFromTensor(&a_scale);
        inputs[B_scale]->CopyDataFromTensor(&b_scale);
        inputs[out_scale]->CopyDataFromTensor(&o_scale);
        AQuant.SetScales({a_scale});
        BQuant.SetScales({b_scale});
        OutQuant.SetScales({o_scale});
      } break;
      case tim::vx::DataType::FLOAT16: {
        Ort::Float16_t a_scale, b_scale, o_scale;
        inputs[A_scale]->CopyDataFromTensor(&a_scale);
        inputs[B_scale]->CopyDataFromTensor(&b_scale);
        inputs[out_scale]->CopyDataFromTensor(&o_scale);
        AQuant.SetScales({static_cast<float>(a_scale)});
        BQuant.SetScales({static_cast<float>(b_scale)});
        OutQuant.SetScales({static_cast<float>(o_scale)});
      } break;
    }

    tim::vx::TensorSpec ASpec(inputs[matrixA]->GetSpec());
    tim::vx::TensorSpec BSpec(inputs[matrixB]->GetSpec());
    tim::vx::TensorSpec OutSpec(outputs[0]->GetSpec());
    ASpec.SetQuantization(AQuant);
    BSpec.SetQuantization(BQuant);
    OutSpec.SetQuantization(OutQuant);
    std::shared_ptr<tim::vx::Tensor> real_A = nullptr;
    if (inputs[matrixA]->GetQuantization().Type() != tim::vx::QuantType::NONE) {
      real_A = inputs[matrixA];
    } else {
      real_A = graph_ep->GetGraph()->CreateTensor(ASpec);
      if(inputs[matrixA]->IsConstTensor()){
        std::vector<T2> const_value(inputs[matrixA]->GetSpec().GetElementNum());
        inputs[matrixA]->CopyDataFromTensor(const_value.data());
        real_A->CopyDataToTensor(const_value.data());
      }
    }
    std::shared_ptr<tim::vx::Tensor> real_B = nullptr;
    if (inputs[matrixB]->GetQuantization().Type() != tim::vx::QuantType::NONE) {
      real_B = inputs[matrixB];
    } else {
      real_B = graph_ep->GetGraph()->CreateTensor(BSpec);
      if(inputs[matrixB]->IsConstTensor()){
        std::vector<T2> const_value(inputs[matrixB]->GetSpec().GetElementNum());
        inputs[matrixB]->CopyDataFromTensor(const_value.data());
        real_B->CopyDataToTensor(const_value.data());
      }
    }

    auto real_out = graph_ep->GetGraph()->CreateTensor(OutSpec);
    if (inputs[matrixB]->GetSpec().GetTensorAttribute() == tim::vx::TensorAttribute::CONSTANT) {
      std::vector<T2> B_data(inputs[matrixB]->GetSpec().GetElementNum());
      inputs[matrixB]->CopyDataFromTensor(B_data.data());
      real_B->CopyDataToTensor(B_data.data());
    }

    graph_ep->UpdateTensorMap(node->InputDefs()[matrixA]->Name(), real_A);
    graph_ep->UpdateTensorMap(node->InputDefs()[matrixB]->Name(), real_B);
    graph_ep->UpdateTensorMap(node->OutputDefs()[0]->Name(), real_out);

    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Matmul>();

    std::vector<NodeArg*> input_defs;
    input_defs.push_back(util::RemoveWrapper(node->InputDefs()[matrixA]));
    input_defs.push_back(util::RemoveWrapper(node->InputDefs()[matrixB]));
    auto node_info = graph_ep->ConstructNodeIO(std::move(op), input_defs,util::RemoveWrapper(node->OutputDefs()));
    graph_ep->GetOps().push_back(node_info);
  }
};

bool QLinearMatMulOpBuilder::HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                                           std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                                           std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                                           const Node* node) {
  LOGS_DEFAULT(INFO) << "Creating QLinearMatmul Op.";
  switch (inputs[A_zero_point]->GetDataType()) {
    case tim::vx::DataType::INT8: {
      switch (inputs[B_zero_point]->GetDataType()) {
        case tim::vx::DataType::INT8: {
          switch (inputs[out_zero_point]->GetDataType()) {
            case tim::vx::DataType::INT8:
              QMatMulImpl<int8_t, int8_t, int8_t>(graph_ep, inputs, outputs, node);
              break;
            case tim::vx::DataType::UINT8:
              QMatMulImpl<int8_t, int8_t, uint8_t>(graph_ep, inputs, outputs, node);
              break;
          }
          break;
        }
        case tim::vx::DataType::UINT8: {
          switch (inputs[out_zero_point]->GetDataType()) {
            case tim::vx::DataType::INT8:
              QMatMulImpl<int8_t, uint8_t, int8_t>(graph_ep, inputs, outputs, node);
              break;
            case tim::vx::DataType::UINT8:
              QMatMulImpl<int8_t, uint8_t, uint8_t>(graph_ep, inputs, outputs, node);
              break;
          }
          break;
        }
      }
      break;
    }
    case tim::vx::DataType::UINT8: {
      switch (inputs[B_zero_point]->GetDataType()) {
        case tim::vx::DataType::INT8: {
          switch (inputs[out_zero_point]->GetDataType()) {
            case tim::vx::DataType::INT8:
              QMatMulImpl<uint8_t, int8_t, int8_t>(graph_ep, inputs, outputs, node);
              break;
            case tim::vx::DataType::UINT8:
              QMatMulImpl<uint8_t, int8_t, uint8_t>(graph_ep, inputs, outputs, node);
              break;
          }
          break;
        }
        case tim::vx::DataType::UINT8: {
          switch (inputs[out_zero_point]->GetDataType()) {
            case tim::vx::DataType::INT8:
              QMatMulImpl<uint8_t, uint8_t, int8_t>(graph_ep, inputs, outputs, node);
              break;
            case tim::vx::DataType::UINT8:
              QMatMulImpl<uint8_t, uint8_t, uint8_t>(graph_ep, inputs, outputs, node);
              break;
          }
          break;
        }
      }
      break;
    }
  }

  return true;
}

}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
