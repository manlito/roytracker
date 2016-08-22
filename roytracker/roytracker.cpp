#include <stdexcept>
#include <roytracker/roytracker.hpp>
#include <opencv2/features2d.hpp>

namespace roytracker {

void RoyTracker::ComputeDescriptors()
{
#pragma omp parallel for
  for (size_t image_index = 0; image_index < images_.size(); ++image_index) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Ptr<cv::Feature2D> feature_detector = cv::AKAZE::create();
    feature_detector->detectAndCompute(images_[image_index], cv::Mat(), keypoints, descriptors);
#pragma omp critical
    {
      keypoints_.push_back(keypoints);
      descriptors.push_back(descriptors);
    }
  }
}

void RoyTracker::Track(std::vector<TrackingResult> &results)
{
  // Ensure the reference image is already set
  if (reference_image_.empty()) {
    throw std::logic_error("Reference image is empty");
  }

  results.clear();

  // Compute descriptors for all images
  ComputeDescriptors();

  // For each tracking target
  for (const auto &tracking_objective: tracking_objectives) {
    // Grab the ROI for the tracking objective
    cv::Rect reference_roi = cv::Rect(tracking_objective.options_.x - tracking_objective.options_.width / 2,
                                      tracking_objective.options_.y - tracking_objective.options_.height / 2,
                                      tracking_objective.options_.width,
                                      tracking_objective.options_.height);
    // Make a selection inside boundaries
    reference_roi.x = std::min(reference_image_.size().width - reference_roi.width - 1, std::max(0, reference_roi.x));
    reference_roi.y = std::min(reference_image_.size().height - reference_roi.height - 1, std::max(0, reference_roi.y));

    cv::Mat reference_mask = cv::Mat::zeros(reference_image_.size(), CV_8UC1);
    reference_mask(reference_roi) = 1;

    TrackingResult result;
    result.reference_patch = reference_image_(reference_roi);

    // Extract the features for the reference image
    std::vector<cv::KeyPoint> reference_keypoints;
    cv::Mat reference_descriptors;
    cv::Ptr<cv::Feature2D> descriptor_extractor;
    descriptor_extractor = cv::AKAZE::create();
    descriptor_extractor->detectAndCompute(reference_image_,
                                           reference_mask,
                                           reference_keypoints,
                                           reference_descriptors);

    // Now match agains non-reference images
//    for (const auto &cv::Mat : images_) {
//      cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
//      std::vector<cv::DMatch> matches;
//      //matcher->match();
//    }

    results.push_back(result);
  }
}

}
