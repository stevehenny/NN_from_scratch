include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)


enable_testing()


add_executable(
  hello_test
  hello_test.cc
)

add_executable(
  cuda_kernel_tests 
  cuda_kernel_tests.cu
)



target_link_libraries(
  hello_test
  GTest::gtest_main
)


target_link_libraries(
  cuda_kernel_tests
  PRIVATE 
  Layers 
  CudaKernels
  GTest::gtest_main
)


include(GoogleTest)
gtest_discover_tests(hello_test)
gtest_discover_tests(cuda_kernel_tests)
