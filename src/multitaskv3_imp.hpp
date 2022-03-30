

#pragma once
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/multitaskv3.hpp>

namespace vitis {
namespace ai {

class MultiTaskv3Imp : public vitis::ai::TConfigurableDpuTask<MultiTaskv3> {
 public:
  MultiTaskv3Imp(const std::string& model_name, bool need_preprocess = true);
  virtual ~MultiTaskv3Imp();

 private:
  virtual MultiTaskv3Result run_8UC1(const cv::Mat& image) override;
  virtual std::vector<MultiTaskv3Result> run_8UC1(
      const std::vector<cv::Mat>& images) override;
  virtual MultiTaskv3Result run_8UC3(const cv::Mat& image) override;
  virtual std::vector<MultiTaskv3Result> run_8UC3(
      const std::vector<cv::Mat>& images) override;

 private:
  void run_it(const cv::Mat& image);
  void run_it(const std::vector<cv::Mat>& images);

 private:
  std::unique_ptr<MultiTaskv3PostProcess> processor_;
};

}  // namespace ai
}  // namespace vitis
