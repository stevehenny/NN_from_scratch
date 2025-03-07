add_library(LoadData STATIC LoadData.cpp)

add_executable(main main.cu)

set_target_properties(main PROPERTIES 
 CUDA_SEPARABLE_COMPILATION ON
 CUDA_STANDARD 17 # Use C++17 for CUDA
)

# Link CUDA libraries
target_link_libraries(main PRIVATE ${CUDAToolkit_LIBRARIES})

# Include CUDA headers
target_include_directories(main PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Add CUDA runtime
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(main PRIVATE LoadData)
