---
title: Verisilicon - VSINPU
description: Instructions to execute ONNX Runtime with VsiNpu
parent: Execution Providers
# nav_order: 15
---
{::options toc_levels="2" /}

# VSINPU Execution Provider
VsiNpu constructed with TIM-VX as an onnxruntime Execution Provider.

[TIM-VX](https://github.com/VeriSilicon/TIM-VX) is a software integration module provided by VeriSilicon to facilitate deployment of Neural-Networks on VeriSilicon ML accelerators. It serves as the backend binding for runtime frameworks such as Android NN, Tensorflow-Lite, MLIR, TVM and more.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Prepare

The VSINPU Execution Provider (EP) requires that tim-vx has already been built building EP.

See [here](https://github.com/VeriSilicon/TIM-VX?tab=readme-ov-file#build-and-run) for building tim-vx-build.

## Build
### Linux

1. export envs
```bash
export TIM_VX_INSTALL=<your-path-to-tim-vx-build>/install
export VIVANTE_SDK_DIR=<your_path_of_tim-vx>/prebuilt-sdk/x86_64_linux
export LD_LIBRARY_PATH=$VIVANTE_SDK_DIR/lib:$TIM_VX_INSTALL/lib
```

2. build onnxruntime with "--use_vsinpu"

```bash
./build.sh --config Debug --build_shared_lib --use_vsinpu --skip_tests
```

## Test
VSINPU EP can run onnx models (unit-tests)with onnx binary onnx_test_runner (onnxruntime_test_all).
1. export envs
```bash
export TIM_VX_INSTALL=<your-path-to-tim-vx-build>/install
export VIVANTE_SDK_DIR=<your_path_of_tim-vx>/prebuilt-sdk/x86_64_linux
export LD_LIBRARY_PATH=$VIVANTE_SDK_DIR/lib:$TIM_VX_INSTALL/lib
```
2. run test
```bash
cd <your_path_to_onnxruntime_build>/Debug/
./onnx_test_runner -e vsinpu <yout_path_to_onnx_model>

cd <your_path_to_onnxruntime_build>/Debug/
./onnxruntime_test_all --gtest_filter=ActivationOpTest.Sigmoid
```
If detailed log information is required, please add the "-v" parameter.

## Supported ops
Following ops are supported by the VSINPU Execution Provider, it should be noted that all operators do not support dynamic shape input or output.

|Operator|Note|
|--------|------|
|ai.onnx:Abs||
|ai.onnx:Add||
|ai.onnx:Conv|Only 1D/2D Conv is supported yet.|
|ai.onnx:Div||
|ai.onnx:Exp||
|ai.onnx:Floor||
|ai.onnx:Gemm||
|ai.onnx:GlobalAveragePool||
|ai.onnx:HardSigmoid||
|ai.onnx:HardSwish||
|ai.onnx:LeakyRelu||
|ai.onnx:Log||
|ai.onnx:Mul||
|ai.onnx:Pow||
|ai.onnx:Relu||
|ai.onnx:Sigmoid||
|ai.onnx:Sin||
|ai.onnx:Sqrt||
|ai.onnx:Sub||
|ai.onnx:Tanh||
