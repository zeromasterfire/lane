#include "vitis/ai/multitaskv3.hpp"

#include "./multitaskv3_imp.hpp"

namespace vitis {
namespace ai {

MultiTaskv3::MultiTaskv3() {}
MultiTaskv3::~MultiTaskv3() {}

std::unique_ptr<MultiTaskv3> MultiTaskv3::create(const std::string& model_name,
                                             bool need_preprocess) {
  return std::unique_ptr<MultiTaskv3>(
      new MultiTaskv3Imp(model_name, need_preprocess));
}

}  // namespace ai
}  // namespace vitis
