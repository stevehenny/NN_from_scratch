cmake_minimum_required(VERSION 3.18) 
project(NN_from_Scratch LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/tests)
enable_language(CUDA)
# Add src as a subdirectory
add_subdirectory(src)
add_subdirectory(tests)
# Fix compile_commands.json for Clangd by removing NVCC entries
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/compile_commands_fixed.json
  COMMAND cat ${CMAKE_BINARY_DIR}/compile_commands.json | grep -v '"command":.*nvcc' > ${CMAKE_BINARY_DIR}/compile_commands_fixed.json
  COMMAND mv ${CMAKE_BINARY_DIR}/compile_commands_fixed.json ${CMAKE_BINARY_DIR}/compile_commands.json
  DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Fixing compile_commands.json for Clangd"
  VERBATIM
)

add_custom_target(fix_clangd ALL DEPENDS ${CMAKE_BINARY_DIR}/compile_commands_fixed.json)
