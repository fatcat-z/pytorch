
#include <ATen/DLConvertor.h>

#include <torch/script.h>
#include <torch/csrc/jit/ir/ir.h>

#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/jit/passes/onnx/nezha_helper.h>

#include <pybind11/embed.h>

#include <unistd.h>
#include <sys/wait.h>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>


static std::shared_ptr<std::string> my_name = nullptr;

torch::Tensor dummy_op(torch::Tensor testData) {
    printf(" ===== from dummy_op =====\n");
    if (my_name == nullptr) {
        my_name = std::make_shared<std::string>("update in dummy_op");
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_node_dims = std::vector<int64_t>{2};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)testData.data_ptr(), testData.nbytes(), input_node_dims.data(), 1);
    printf("Yes, got new values.\n");


    // auto input_data = at::toDLPack(testData);
    // auto ort_input_data = onnxruntime::python::OrtValueToDlpack(input_data);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dummy_op");
    Ort::SessionOptions session_options;
    Ort::Session session = Ort::Session(env, "/home/jay/my_nezha_test.onnx", session_options);
    // auto output_tensors = session.Run(session.GetInputNames(), input_tensors, session.GetOutputNames());

    size_t num_input_nodes = session.GetInputCount();
    printf("total input count: %ld\n", num_input_nodes);
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*> input_node_names(num_input_nodes);

    for (int i = 0; i < num_input_nodes; i++) {
        // print input node names
        char* input_name = session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        // input_node_dims = tensor_info.GetShape();
        // printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        // for (int j = 0; j < input_node_dims.size(); j++)
        // printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }


    std::vector<const char*> output_node_names = {"output.13"};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);

    printf(" ===== init Ort::Session successfully in dummy_op. =====\n");
    return testData.clone();
}

torch::Tensor fake_op(torch::Tensor testData, int64_t value) {
    printf(" ===== from fake_op =====\n");
    my_name = std::make_shared<std::string>("update in fake_op");
    auto tempValue = torch::ones_like(testData);
    auto output = torch::add(testData, tempValue);
    return output.clone();
}

torch::Tensor ort_inference_op(std::string file_name, torch::Tensor inputs) {
    torch::Tensor final_output;

    PyEval_InitThreads();
    printf(" ===== from ort_inference_ops:init =====\n");
    pybind11::gil_scoped_acquire acquire;

    py::object py_onnx = py::module::import("torch.onnx");
    printf(" ===== ort_inference_ops: import successfully =====\n");
    py::object my_output = py_onnx.attr("try_ort_inference")(file_name, inputs);
    printf(" ===== ort_inference_ops: finish running method. =====\n");        
    final_output = py::cast<torch::autograd::Variable>(my_output);    
    pybind11::gil_scoped_release release;
    return final_output;
}

std::string get_str_op(torch::Tensor testData) {
    return (*my_name.get() + testData.name());
}

TORCH_LIBRARY(nezha_ops, m) {
  m.def("dummy_op", dummy_op);
  m.def("fake_op", fake_op);
  m.def("ort_inference_op", ort_inference_op);
  m.def("get_str_op", get_str_op);
}