
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(nezha_ops)

find_package(Torch REQUIRED)

# Define our library target
add_library(nezha_ops SHARED nezha_ops.cpp)
# Enable C++14
target_compile_features(nezha_ops PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(nezha_ops "${TORCH_LIBRARIES}")
