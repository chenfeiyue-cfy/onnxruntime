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
#include "core/framework/compute_capability.h"
#include "vsinpu_execution_provider.h"
#include "vsinpu_ep_graph.h"
#include "builders/op_builder_factory.h"
#include "builders/op_builder.h"
#include "core/framework/kernel_registry.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

VSINPUExecutionProvider::VSINPUExecutionProvider(const VSINPUExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kVSINPUExecutionProvider}, device_id_(info.device_id) {
  AllocatorCreationInfo default_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo("VSINPU", OrtAllocatorType::OrtDeviceAllocator));
      }};

  CreateAllocator(default_memory_info);

  AllocatorCreationInfo cpu_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo("VSINPU", OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      }};

  CreateAllocator(cpu_memory_info);
}

VSINPUExecutionProvider::~VSINPUExecutionProvider() {}

// If not partitioned, return a complete graph; else return a vector of subgraphs
static void AppendClusterToSubGraph(const std::vector<NodeIndex>& nodes,
                                    const std::vector<std::string>& onnx_inputs,
                                    const std::vector<std::string>& onnx_outputs,
                                    std::vector<std::unique_ptr<ComputeCapability>>& result) {
  static size_t op_counter = 0;

  auto meta_def = std::make_unique<IndexedSubGraph::MetaDef>();
  meta_def->name = "VSINPUOp_" + std::to_string(++op_counter);
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->inputs = onnx_inputs;
  meta_def->outputs = onnx_outputs;

  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
  sub_graph->nodes = nodes;
  sub_graph->SetMetaDef(std::move(meta_def));
  result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
}

// return a vector of unsupported nodes' index
static std::vector<NodeIndex> GetUnsupportedNodeIndices(
    const GraphViewer& graph_viewer,
    /*out*/ std::unordered_set<std::string>& vsinpu_required_initializers) {
  std::vector<NodeIndex> unsupported_nodes_idx;

  for (const auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    auto node = graph_viewer.GetNode(node_idx);
    if (vsi::npu::GraphEP::SupportedOp(graph_viewer, node)) {
      // Collect inputs that are initializers
      LOGS_DEFAULT(VERBOSE) << "node:" << node->OpType();
      node->ForEachDef(
          [&vsinpu_required_initializers, &graph_viewer](
              const onnxruntime::NodeArg& node_arg, bool is_input) {
            if (is_input &&
                graph_viewer.GetAllInitializedTensors().count(node_arg.Name())) {
              vsinpu_required_initializers.insert(node_arg.Name());
              LOGS_DEFAULT(VERBOSE) << "Input tensor:" << vsi::npu::util::PrintNode(node_arg);
            }
          },
          true);
    } else {
      LOGS_DEFAULT(WARNING) << "Unsupported node:" << node->OpType();
      unsupported_nodes_idx.push_back(node_idx);
    }
  }

  return unsupported_nodes_idx;
}

/**
 * Returns a vector clusters(or node_idx). For each unsupported node, the graph is split into 3
 * parts. supported_cluster + (UNsupported_node + rest_of_the_graph). This functions returns vector
 * of all supported_clusters by VSINPU
 */
static std::vector<std::vector<NodeIndex>> GetPartitionedClusters(
    const std::vector<NodeIndex>& topological_order,
    const std::vector<NodeIndex>& unsupported_nodes) {
  std::vector<std::vector<NodeIndex>> vsinpu_clusters;

  auto prev = topological_order.begin();

  for (const auto& unsup_node : unsupported_nodes) {
    auto it = std::find(prev, topological_order.end(), unsup_node);
    // Create a cluster vector[supported_node_idx, unsupported_node_idx) and append it to return
    // list.
    std::vector<NodeIndex> this_cluster{prev, it};
    if (!this_cluster.empty()) {
      vsinpu_clusters.push_back(std::move(this_cluster));
    }
    // Point prev to node idx past this unsuported node.
    prev = ++it;
  }

  // Tail
  std::vector<NodeIndex> this_cluster{prev, topological_order.end()};
  if (!this_cluster.empty()) {
    vsinpu_clusters.push_back(std::move(this_cluster));
  }

  return vsinpu_clusters;
}

// sort out every cluster's input/output
static void GetIOofCluster(
    const GraphViewer& graph_viewer,
    const std::vector<NodeIndex>& cluster,
    const std::unordered_set<std::string>& vsinpu_required_initializers,
    /*out*/ std::vector<std::string>& cluster_inputs,
    /*out*/ std::vector<std::string>& cluster_outputs) {
  std::unordered_set<std::string> input_args;
  std::vector<std::string> ordered_input_args;
  std::unordered_set<std::string> output_args;
  std::unordered_set<std::string> external_output_args;

  for (const auto& node_idx : cluster) {
    const auto& node = graph_viewer.GetNode(node_idx);

    node->ForEachDef(
        [&input_args, &ordered_input_args, &output_args](const NodeArg& node_arg,
                                                         bool is_input) {
          if (is_input) {
            if (!input_args.count(node_arg.Name())) {
              ordered_input_args.push_back(node_arg.Name());
            }
            input_args.insert(node_arg.Name());
          } else {
            output_args.insert(node_arg.Name());
          }
        },
        true);

    // Check if output of this node is used by nodes not in this_cluster. If yes add this to
    // cluster outputs
    for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
      const auto& ext_node = graph_viewer.GetNode((*it).Index());

      if (std::find(cluster.begin(), cluster.end(), ext_node->Index()) == cluster.end()) {
        // Node is external to this_cluster. Search through its inputs to find the output
        // that is generated by this_cluster.
        std::set<std::string> ext_node_inputs;
        ext_node->ForEachDef(
            [&ext_node_inputs](const onnxruntime::NodeArg& arg, bool is_input) {
              if (is_input) {
                ext_node_inputs.insert(arg.Name());
              }
            },
            true);

        for (const auto& out_def : node->OutputDefs()) {
          if (ext_node_inputs.find(out_def->Name()) != ext_node_inputs.end()) {
            external_output_args.insert(out_def->Name());
          }
        }
      }
    }
  }  // end processing one cluster

  // Extract initializers used by this_cluster.
  std::unordered_set<std::string> original_graph_inputs;
  for (const auto& node_arg : graph_viewer.GetInputsIncludingInitializers()) {
    original_graph_inputs.insert(node_arg->Name());
  }

  const auto& initializers = graph_viewer.GetAllInitializedTensors();
  std::vector<std::string> const_inputs;
  for (const auto& in_arg : ordered_input_args) {
    if ((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
        vsinpu_required_initializers.count(in_arg)) {
      const_inputs.push_back(in_arg);
    }
  }

  for (const auto& in_arg : ordered_input_args) {
    if (!output_args.count(in_arg) &&
        !((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
          vsinpu_required_initializers.count(in_arg))) {
      cluster_inputs.push_back(in_arg);
    }
  }

  for (const auto& in_arg : const_inputs) {
    cluster_inputs.push_back(in_arg);
  }

  std::copy(external_output_args.begin(),
            external_output_args.end(),
            std::back_inserter(cluster_outputs));
  for (const auto& node_arg : graph_viewer.GetOutputs()) {
    const auto& name = node_arg->Name();
    if (output_args.count(name) && !external_output_args.count(name)) {
      cluster_outputs.push_back(name);
    }
  }
}

std::vector<std::unique_ptr<ComputeCapability>> VSINPUExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const IKernelLookup& /*kernel_lookup*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  for (const auto& tensor : graph_viewer.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location()) {
      LOGS_DEFAULT(VERBOSE) << "location:" << tensor.second->data_location();
      if (tensor.second->data_location() ==
          ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
        LOGS_DEFAULT(WARNING) << "VSINPU: Initializers with external data location are not "
                                 "currently supported";
        return result;
      }
    }
  }

  /* This is a list of initializers that nGraph considers as constants. Example weights, reshape
     shape etc.
     TODO: Support overridable initializers */
  std::unordered_set<std::string> vsinpu_required_initializers;
  const auto unsupported_nodes =
      GetUnsupportedNodeIndices(graph_viewer, vsinpu_required_initializers);

  // If all ops are supported, no partitioning is required. Short-circuit and avoid splitting.
  if (unsupported_nodes.empty()) {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    // Fill inputs with names
    std::for_each(graph_viewer.GetInputs().begin(),
                  graph_viewer.GetInputs().end(),
                  [&inputs](const NodeArg* node_arg) { inputs.push_back(node_arg->Name()); });

    /* In scenarios, when there are no inputs or all inputs being initializers,
         ConstantFolding optimization in onnxruntime pre-computes the value.*/
    if (inputs.empty()) {
      return result;
    }

    // Initializers need to be part of meta_def->inputs
    std::for_each(vsinpu_required_initializers.begin(),
                  vsinpu_required_initializers.end(),
                  [&inputs](const std::string& initializer) { inputs.push_back(initializer); });

    // Fill outputs with names
    std::for_each(graph_viewer.GetOutputs().begin(),
                  graph_viewer.GetOutputs().end(),
                  [&outputs](const NodeArg* node_arg) { outputs.push_back(node_arg->Name()); });

    // Create and add this graph to result.
    AppendClusterToSubGraph(graph_viewer.GetNodesInTopologicalOrder(), inputs, outputs, result);

  } else {
    const auto vsinpu_clusters =
        GetPartitionedClusters(graph_viewer.GetNodesInTopologicalOrder(), unsupported_nodes);

    for (const auto& this_cluster : vsinpu_clusters) {
      std::vector<std::string> cluster_inputs, cluster_outputs;
      GetIOofCluster(graph_viewer,
                     this_cluster,
                     vsinpu_required_initializers,
                     cluster_inputs,
                     cluster_outputs);

      if (!cluster_inputs.empty()) {
        AppendClusterToSubGraph(this_cluster, cluster_inputs, cluster_outputs, result);
      }
    }
  }

  return result;
}

Status ComputeStateFunc(vsi::npu::GraphEP* graph_ep,
                        OrtKernelContext* context) {
  Ort::KernelContext ctx(context);
  const size_t num_inputs = graph_ep->GetGraphInputs().size();

  for (size_t i = 0, j = 0; i < num_inputs; i++) {
    if (!graph_ep->GetGraphInputs()[i]->is_initializer) {
      const auto onnx_input_tensor = ctx.GetInput(i);
      const auto tensor_info = onnx_input_tensor.GetTensorTypeAndShapeInfo();

      auto origin_tensor = graph_ep->GetGraphInputs()[i]->tensor;
      origin_tensor->CopyDataToTensor(onnx_input_tensor.GetTensorRawData(), vsi::npu::util::GetTensorBytes(tensor_info));
      j++;
    }
  }

  if (!graph_ep->GetGraph()->Run()) {
    LOGS_DEFAULT(ERROR) << "Failed to run graph.";
  }
  for (size_t i = 0; i < ctx.GetOutputCount(); i++) {
    auto timvx_tensor = graph_ep->GetGraphOutputs()[i]->tensor;
    auto out_shape = graph_ep->GetGraphOutputs()[i]->shape.GetDims();
    auto onnx_output_tensor =
        ctx.GetOutput(i, out_shape.data(), out_shape.size());
    timvx_tensor->CopyDataFromTensor((void*)onnx_output_tensor.GetTensorRawData());
  }

  return Status::OK();
}

Status VSINPUExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                        std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_viewer = fused_node_graph.filtered_graph;
    NodeComputeInfo compute_info;
    std::shared_ptr<vsi::npu::GraphEP> graph_ep = std::make_shared<vsi::npu::GraphEP>();

    for (auto tensor : graph_viewer.GetInputsIncludingInitializers()) {
      LOGS_DEFAULT(VERBOSE) << "subgraph input init:" << vsi::npu::util::PrintNode(*tensor) << "#"
                            << graph_viewer.IsInitializedTensor(tensor->Name());
      auto input = std::make_shared<vsi::npu::GraphIOInfo>();
      input->name = tensor->Name();
      if (graph_viewer.IsInitializedTensor(tensor->Name())) {
        input->is_initializer = true;
      } else {
        input->is_initializer = false;
      }
      graph_ep->GetGraphInputs().push_back(input);
    }
    for (auto tensor : graph_viewer.GetOutputs()) {
      LOGS_DEFAULT(VERBOSE) << "subgraph output:" << vsi::npu::util::PrintNode(*tensor);
      auto output = std::make_shared<vsi::npu::GraphIOInfo>();
      output->name = tensor->Name();
      output->is_initializer = false;
      graph_ep->GetGraphOutputs().push_back(output);
    }

    for (const auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
      auto node = graph_viewer.GetNode(node_index);
      LOGS_DEFAULT(VERBOSE) << "sub node:" << node->OpType();
      vsi::npu::SupportedBuiltinOps().at(node->OpType())->BuildOp(graph_ep.get(), graph_viewer, node);
    }

    LOGS_DEFAULT(INFO) << "Verifying graph";
    graph_ep->GetCompiled() = graph_ep->GetGraph()->Compile();
    if (!graph_ep->GetCompiled()) {
      LOGS_DEFAULT(ERROR) << "Failed to verify graph.";
    } else
      LOGS_DEFAULT(INFO) << "Graph has been verified successfully.";

    compute_info.create_state_func = [graph_ep](ComputeContext* /*context*/,
                                                FunctionState* state) {
      *state = graph_ep.get();
      return 0;
    };

    compute_info.compute_func =
        [graph_ep, this](FunctionState state, const OrtApi* /* api */,
                         OrtKernelContext* context) {
          std::lock_guard<OrtMutex> lock(this->GetMutex());
          Status res = ComputeStateFunc(graph_ep.get(), context);
          return res;
        };

    compute_info.release_state_func = [](FunctionState /*state*/) {};

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> VSINPUExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  return kernel_registry;
}

}  // namespace onnxruntime
