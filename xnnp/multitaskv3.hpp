

#pragma once
#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <opencv2/core.hpp>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {


struct Vehiclev3Result {
  /// number of classes
  /// \li \c  0 : "car"
  /// \li \c 1 : "sign"
  /// \li \c  2 : "person"
  int label;
  float score;
  float x;
  float y;
  float width;
  float height;
  float angle;
};

struct MultiTaskv3Result {
  int width;
  int height;
  std::vector<Vehiclev3Result> vehicle;
  cv::Mat segmentation;
  cv::Mat lane;
  cv::Mat drivable;
  cv::Mat depth;
};

class MultiTaskv3PostProcess {
 public:
  static std::unique_ptr<MultiTaskv3PostProcess> create(
      const std::vector<std::vector<vitis::ai::library::InputTensor>>&
          input_tensors,
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors,
      const vitis::ai::proto::DpuModelParam& config);
 protected:
  explicit MultiTaskv3PostProcess();
  MultiTaskv3PostProcess(const MultiTaskv3PostProcess&) = delete;
  MultiTaskv3PostProcess& operator=(const MultiTaskv3PostProcess&) = delete;

 public:
  virtual ~MultiTaskv3PostProcess();
  virtual std::vector<MultiTaskv3Result> post_process(size_t batch_size) = 0;
  virtual std::vector<MultiTaskv3Result> post_process_visualization(
      size_t batch_size) = 0;
};

}  // namespace ai
}  // namespace vitis
