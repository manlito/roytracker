#include <stdexcept>
#include <roytracker/roytracker.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace roytracker {

void RoyTracker::ComputeDescriptors()
{
  keypoints_.resize(images_.size());
  descriptors_.resize(images_.size());
#pragma omp parallel for
  for (size_t image_index = 0; image_index < images_.size(); ++image_index) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Ptr<cv::Feature2D> feature_detector = cv::AKAZE::create();
    feature_detector->detectAndCompute(images_[image_index], cv::Mat(), keypoints, descriptors);
#pragma omp critical
    {
      keypoints_[image_index] = keypoints;
      descriptors_[image_index] = descriptors;
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

    TrackingResult result;
    result.reference_patch = reference_image_(reference_roi);

    // Extract the features for the reference image
    std::vector<cv::KeyPoint> reference_keypoints;
    cv::Mat reference_descriptors;
    cv::Ptr<cv::Feature2D> descriptor_extractor;
    descriptor_extractor = cv::AKAZE::create();
    descriptor_extractor->detectAndCompute(reference_image_,
                                           cv::Mat(),
                                           reference_keypoints,
                                           reference_descriptors);

    // Now match agains non-reference images
#pragma omp parallel for
    for (size_t image_index = 0; image_index < images_.size(); ++image_index) {

      // Matcher
      std::vector<cv::DMatch> matches;
      cv::BFMatcher matcher(cv::NORM_L1, false);
      matcher.match(reference_descriptors, descriptors_[image_index], matches);

      // Computation of fundamental matrix to obtain inliers
      cv::Mat inliers_mask;
      {
        std::vector<cv::Point2d> source_points, target_points;
        for (const cv::DMatch &match: matches) {
          const cv::KeyPoint keypoint_reference = reference_keypoints[match.queryIdx];
          const cv::KeyPoint keypoint_image = keypoints_[image_index][match.trainIdx];
          source_points.push_back(keypoint_reference.pt);
          target_points.push_back(keypoint_image.pt);
        }
        cv::findFundamentalMat(source_points, target_points, cv::FM_RANSAC, 5, 0.96, inliers_mask);
      }

      // Computation of median in X and Y
      std::vector<cv::KeyPoint> reference_keypoints_in_roi;
      std::vector<cv::KeyPoint> keypoints_image;
      cv::Point2d center(0, 0);
      {
        std::vector<double> x_coordinates, y_coordinates;
        size_t keypoint_index = 0;
        for (const cv::DMatch &match: matches) {
          const cv::KeyPoint keypoint_reference = reference_keypoints[match.queryIdx];
          const cv::KeyPoint keypoint_image = keypoints_[image_index][match.trainIdx];
          if (inliers_mask.at<unsigned char>(keypoint_index)) {
            if (keypoint_reference.pt.x >= reference_roi.x &&
                keypoint_reference.pt.x <= reference_roi.x + reference_roi.width &&
                keypoint_reference.pt.y >= reference_roi.y &&
                keypoint_reference.pt.y <= reference_roi.y + reference_roi.height) {
              reference_keypoints_in_roi.push_back(keypoint_reference);
              keypoints_image.push_back(keypoint_image);
              x_coordinates.push_back(keypoint_image.pt.x);
              y_coordinates.push_back(keypoint_image.pt.y);
            }
          }
          ++keypoint_index;
        }
        if (x_coordinates.size() == 0) {
          // image did not produce enough matches
          continue;
        }
        size_t middle_element = x_coordinates.size() / 2;
        std::nth_element(x_coordinates.begin(), x_coordinates.begin() + middle_element, x_coordinates.end());
        std::nth_element(y_coordinates.begin(), y_coordinates.begin() + middle_element, y_coordinates.end());
        center.x = x_coordinates[middle_element];
        center.y = y_coordinates[middle_element];
      }

      // Report result
      {
        PatchResult patch_result;
        cv::Rect result_roi = cv::Rect(center.x - tracking_objective.options_.width / 2,
                                       center.y - tracking_objective.options_.height / 2,
                                       tracking_objective.options_.width,
                                       tracking_objective.options_.height);
        result_roi.x = std::min(reference_image_.size().width  - result_roi.width - 1, std::max(0, result_roi.x));
        result_roi.y = std::min(reference_image_.size().height - result_roi.height - 1, std::max(0, result_roi.y));
        patch_result.first = images_[image_index](result_roi);
//        cv::Mat out = images_[image_index].clone();
//        cv::circle(out, centroid, 40, cv::Scalar(0, 240, 0), 2);
//        for (const cv::DMatch &match: matches) {
//          const cv::KeyPoint keypoint = keypoints_[image_index][match.trainIdx];
//          cv::circle(out, keypoint.pt, 3, cv::Scalar(128, 255, 0), 1);
//        }
//        result.reference_patch = reference_image_.clone();
//        for (auto keypoint : reference_keypoints_in_roi) {
//          cv::circle(result.reference_patch, keypoint.pt, 3, cv::Scalar(128, 255, 0), 1);
//        }
//        for (auto keypoint : keypoints_image) {
//          cv::circle(out, keypoint.pt, 3, cv::Scalar(128, 255, 0), 1);
//        }
//        patch_result.first = out;
#pragma omp critical
        result.resulting_patches.push_back(patch_result);
      }
    }

    results.push_back(result);
  }
}

}
