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

#pragma once
#include "core/framework/op_kernel.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
namespace util {

tim::vx::DataType OnnxDtypeToTIMVXDtype(const int32_t dtype);

tim::vx::DataType OnnxDtypeToTIMVXDtype(const ONNX_NAMESPACE::DataType type);

tim::vx::ShapeType OnnxShapeToTIMVXShape(const onnxruntime::TensorShape);

std::string PrintNode(const onnxruntime::NodeArg& node_arg);

std::string PrintNode(const std::vector<int64_t> shape);

size_t GetTensorElementSize(const ONNXTensorElementDataType type);

size_t GetTensorBytes(const Ort::TensorTypeAndShapeInfo& info);

TensorShape GetTensorShape(const onnxruntime::NodeArg& node_arg);

void SetTensorDims(const onnxruntime::NodeArg& node_arg,
                   std::vector<uint32_t>& dims);

std::shared_ptr<uint8_t> UnpackTensor(
    const NodeArg* node, const ONNX_NAMESPACE::TensorProto& initializer);

tim::vx::PadType GetPadType(const std::string type);

bool CheckMainInputType(const Node* node, std::string& reason);

bool CheckZeroDim(const NodeArg* node_arg);

bool ExcludeType(const NodeArg* node_arg, std::string& reason);

bool CheckAllExcludeType(const Node* node, std::string& reason);

bool CheckNoZeroDim(const Node* node);

int32_t ReverseAxis(int32_t origin_axis, int32_t length);

std::vector<int32_t> ReverseAxis(std::vector<int32_t> origin_axes, int32_t length);

}  // namespace util
}  // namespace npu
}  // namespace vsi
}  // namespace onnxruntime
