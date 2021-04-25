#pragma once

#include <torch/script.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

std::vector<Module> NeZha_GetSplitModules(
    Module& module);

void NeZha_TryUpdateModule(
    Module& dst_module,
    std::shared_ptr<Graph>& src_graph);

Module NeZha_UpdateOps(
    Module& dst_module);

Module NeZha_ConvertModule(Module& module, torch::Tensor input);


} // namespace jit

} // namespace torch
