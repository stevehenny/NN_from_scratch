cmake_minimum_required(VERSION 3.18) 
project(NN_from_Scratch LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)
# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
# Add src as a subdirectory
add_subdirectory(src)
