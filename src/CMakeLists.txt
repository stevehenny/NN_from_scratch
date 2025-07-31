# Create static library for data loading
add_library(LoadData STATIC LoadData.cpp)

add_library(Layers STATIC
  LayerClasses.cu Network.cu
)

add_library(CudaKernels STATIC cudaKernels.cu)

target_include_directories(Layers PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(Layers PRIVATE ${CUDAToolkit_LIBRARIES})
set_target_properties(Layers PROPERTIES CUDA_STANDARD 17)
# Compile the CUDA executable

add_executable(main main.cu )
target_link_libraries(main PRIVATE Layers CudaKernels LoadData)

add_executable(mlp mlp.cu)
target_link_libraries(mlp PRIVATE Layers CudaKernels LoadData)

# Set CUDA compilation properties
set_target_properties(main PROPERTIES 
  CUDA_SEPARABLE_COMPILATION OFF  # Turn OFF relocatable device code to avoid -rdc=true issue
  CUDA_STANDARD 17                # Use C++17 for CUDA
)
target_compile_features(main PUBLIC cuda_std_17)
# Link CUDA libraries
target_link_libraries(main PRIVATE ${CUDAToolkit_LIBRARIES})

# Include CUDA headers
target_include_directories(main PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Add NVCC-specific options while ensuring Clangd doesn't see them
# target_compile_options(main PRIVATE 
#   $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
# )

