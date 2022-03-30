
vai_add_library(
  SRCS include/vitis/ai/multitaskv3.hpp src/multitaskv3.cpp src/multitaskv3_imp.cpp
  PUBLIC_REQUIRE ${PROJECT_NAME}::xnnpp ${PROJECT_NAME}::dpu_task)

vai_add_test(test_multitaskv3)
vai_add_test(test_multitaskv3_batch)
