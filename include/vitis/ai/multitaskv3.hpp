#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <xnnp/multitaskv3.hpp>

namespace vitis {
namespace ai {

class MultiTaskv3 {
 public:
  static std::unique_ptr<MultiTaskv3> create(const std::string& model_name,
                                           bool need_preprocess = true);
 protected:
  explicit MultiTaskv3();
  MultiTaskv3(const MultiTaskv3&) = delete;

 public:
  virtual ~MultiTaskv3();
 public:
  virtual int getInputWidth() const = 0;
  virtual int getInputHeight() const = 0;
  virtual size_t get_input_batch() const = 0;
  virtual MultiTaskv3Result run_8UC1(const cv::Mat& image) = 0;
  virtual std::vector<MultiTaskv3Result> run_8UC1(
      const std::vector<cv::Mat>& images) = 0;

  virtual MultiTaskv3Result run_8UC3(const cv::Mat& image) = 0;
  virtual std::vector<MultiTaskv3Result> run_8UC3(
      const std::vector<cv::Mat>& images) = 0;
};

class MultiTaskv38UC1 {
 public:
  static std::unique_ptr<MultiTaskv38UC1> create(const std::string& model_name,
                                               bool need_preprocess = true) {
    return std::unique_ptr<MultiTaskv38UC1>(
        new MultiTaskv38UC1(MultiTaskv3::create(model_name, need_preprocess)));
  }
 protected:
  explicit MultiTaskv38UC1(std::unique_ptr<MultiTaskv3> multitaskv3)
      : multitaskv3_{std::move(multitaskv3)} {}
  MultiTaskv38UC1(const MultiTaskv38UC1&) = delete;

 public:
  virtual ~MultiTaskv38UC1() {}
 public:
  virtual int getInputWidth() const { return multitaskv3_->getInputWidth(); }
  virtual int getInputHeight() const { return multitaskv3_->getInputHeight(); }

  virtual size_t get_input_batch() const {
    return multitaskv3_->get_input_batch();
  }

  
  virtual MultiTaskv3Result run(const cv::Mat& image) {
    return multitaskv3_->run_8UC1(image);
  }
  virtual std::vector<MultiTaskv3Result> run(const std::vector<cv::Mat>& images) {
    return multitaskv3_->run_8UC1(images);
  }
  
 private:
  std::unique_ptr<MultiTaskv3> multitaskv3_;
 
};

class MultiTaskv38UC3 {
 public:
  static std::unique_ptr<MultiTaskv38UC3> create(const std::string& model_name,
                                               bool need_preprocess = true) {
    return std::unique_ptr<MultiTaskv38UC3>(
        new MultiTaskv38UC3(MultiTaskv3::create(model_name, need_preprocess)));
  }
 protected:
  explicit MultiTaskv38UC3(std::unique_ptr<MultiTaskv3> multitaskv3)
      : multitaskv3_{std::move(multitaskv3)} {}
  MultiTaskv38UC3(const MultiTaskv38UC3&) = delete;

 public:
  virtual ~MultiTaskv38UC3() {}
 
 public:

  virtual int getInputWidth() const { return multitaskv3_->getInputWidth(); }

  virtual int getInputHeight() const { return multitaskv3_->getInputHeight(); }
  virtual size_t get_input_batch() const {
    return multitaskv3_->get_input_batch();
  }

  virtual MultiTaskv3Result run(const cv::Mat& image) {
    return multitaskv3_->run_8UC3(image);
  }
  virtual std::vector<MultiTaskv3Result> run(const std::vector<cv::Mat>& images) {
    return multitaskv3_->run_8UC3(images);
  }

 private:
  std::unique_ptr<MultiTaskv3> multitaskv3_;

};

}  // namespace ai
}  // namespace vitis
