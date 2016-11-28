#include <stdexcept>
#include <iostream>
#include <roytracker/roytracker.hpp>
#include <roytracker/trackers/mean_shift_tracker.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

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

cv::Mat RoyTracker::ReadImage(std::string filename, bool normalize)
{
  cv::Mat image = cv::imread(filename);
  if (image.channels() >= 3 && normalize) {
    cv::Mat ycrcb_image;

    cv::cvtColor(image, ycrcb_image, CV_BGR2YCrCb);

    std::vector<cv::Mat> channels;
    cv::split(ycrcb_image,channels);

    cv::equalizeHist(channels[0], channels[0]);

    cv::Mat equalized_image;
    cv::merge(channels, ycrcb_image);

    cv::cvtColor(ycrcb_image, equalized_image, CV_YCrCb2BGR);

    return equalized_image;
  }
  return image;
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
    for (size_t image_index = 0; image_index < images_.size(); ++image_index) {

      // Image accesor
      const cv::Mat &image = images_[image_index];

      // Matcher
      std::vector<std::vector<cv::DMatch>> matches_groups;
      std::vector<cv::DMatch> matches;
      cv::BFMatcher matcher(cv::NORM_L1, false);
      matcher.knnMatch(reference_descriptors, descriptors_[image_index], matches_groups, 2);

      // Apply filter of distance to second best match
      for (size_t i = 0; i < matches_groups.size(); i++) {
          if (matches_groups[i][0].distance < 0.8 * matches_groups[i][1].distance) {
            matches.push_back(matches_groups[i][0]);
          }
      }

      // Computation of fundamental matrix to obtain inliers
      cv::Mat inliers_mask;
      {
        std::vector<cv::Point2d> source_points, target_points;
        for (const cv::DMatch &match : matches) {
          const cv::KeyPoint keypoint_reference = reference_keypoints[match.queryIdx];
          const cv::KeyPoint keypoint_image = keypoints_[image_index][match.trainIdx];
          source_points.push_back(keypoint_reference.pt);
          target_points.push_back(keypoint_image.pt);
        }
        cv::Mat f = cv::findFundamentalMat(source_points, target_points, inliers_mask, cv::FM_RANSAC, 5, 0.95);
      }

      size_t inlier_count = 0;
      for (size_t i = 0; i < inliers_mask.rows; ++i) {
        if (inliers_mask.at<unsigned char>(i)) {
          ++inlier_count;
        }
      }

      cv::Rect result_roi;
      if (false && inlier_count >  40) {
        std::cout << "Found a geometric model. Rematching using epipolar constraint" << std::endl;

        // For each keypoint inside the ROI
        std::cout << std::endl << "Inliers: " << inlier_count << " / " << matches.size() << std::endl;
        std::vector<cv::DMatch> matches_filtered;
        for (size_t i = 0; i < inliers_mask.rows; ++i) {
          if (inliers_mask.at<unsigned char>(i)) {
            matches_filtered.push_back(matches[i]);
          }
        }
        cv::Mat matches_image;
        cv::drawMatches(reference_image_, reference_keypoints, image, keypoints_[image_index], matches_filtered, matches_image);
        cv::imwrite("/home/lito/DATA/Datasets/Roytracker/matches_image.jpg", matches_image);
        cv::drawMatches(reference_image_, reference_keypoints, image, keypoints_[image_index], matches, matches_image);
        cv::imwrite("/home/lito/DATA/Datasets/Roytracker/matches_image_full.jpg", matches_image);

        // Computation of median in X and Y
        std::vector<cv::KeyPoint> reference_keypoints_in_roi;
        std::vector<cv::KeyPoint> keypoints_image;
        cv::Point2f center(0, 0);
        {
          std::vector<double> x_coordinates, y_coordinates;
          size_t keypoint_index = 0;
          for (const cv::DMatch &match: matches) {
            const cv::KeyPoint keypoint_reference = reference_keypoints[match.queryIdx];
            const cv::KeyPoint keypoint_image = keypoints_[image_index][match.trainIdx];
            if (1 || inliers_mask.at<unsigned char>(keypoint_index)) {
  //            if (keypoint_reference.pt.x >= reference_roi.x &&
  //                keypoint_reference.pt.x <= reference_roi.x + reference_roi.width &&
  //                keypoint_reference.pt.y >= reference_roi.y &&
  //                keypoint_reference.pt.y <= reference_roi.y + reference_roi.height) {
              {
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
        result_roi = cv::Rect(center.x - tracking_objective.options_.width / 2,
                              center.y - tracking_objective.options_.height / 2,
                              tracking_objective.options_.width,
                              tracking_objective.options_.height);

        // Use Meah-Shift to refine the result
        if (false) {
          MeanShiftTracker mean_shift_tracker;
          mean_shift_tracker.setTargetObject(reference_image_, result_roi);
          mean_shift_tracker.trackObject(image);
          center = mean_shift_tracker.getObjectCenter();
        }

      } else {
        // Discretize image and run MeanShift from a fixed number of quatized starting positions
        std::vector<cv::Point2f> test_positions;
        size_t min_x = reference_roi.width;
        size_t max_x = image.cols - min_x;
        size_t min_y = reference_roi.height;
        size_t max_y = image.rows - min_y;
        double delta_x = (max_x - min_x) / (image.cols / 50.0);
        double delta_y = (max_y - min_y) / (image.rows / 50.0);
        {
          size_t segment_y = 0;
          while (segment_y * delta_y + min_y < max_y) {
            size_t segment_x = 0;
            while (segment_x * delta_x + min_x < max_x) {
              test_positions.push_back(cv::Point2f(segment_x * delta_x + min_x, segment_y * delta_y + min_y));
              ++segment_x;
            }
            ++segment_y;
          }
        }
        std::vector<double> similarity_measurements(test_positions.size());
        std::vector<cv::Point2f> centers(test_positions.size());
#pragma omp parallel for
        for (size_t test_index = 0; test_index < test_positions.size(); ++test_index) {
          MeanShiftTracker mean_shift_tracker;
          mean_shift_tracker.enableWindowScaling = true;
          mean_shift_tracker.setTargetObject(reference_image_, reference_roi);
          mean_shift_tracker.setTargetObjectCenter(test_positions[test_index]);
          mean_shift_tracker.trackObject(image);
#pragma omp critical
          {
            similarity_measurements[test_index] = mean_shift_tracker.lastError;
            centers[test_index] = mean_shift_tracker.getObjectCenter();
            std::cout << "Test (" << test_positions[test_index].x << " " << test_positions[test_index].y
                      << " got " << similarity_measurements[test_index]
                      << " moved to " << centers[test_index].x << " " << centers[test_index].y << std::endl;
          }
        }
        // Find max similarity
        auto max_element = std::max_element(similarity_measurements.begin(), similarity_measurements.end());
        auto max_index = max_element - similarity_measurements.begin();
        result_roi = reference_roi;
        result_roi.x = centers[max_index].x - reference_roi.width / 2;
        result_roi.y = centers[max_index].y - reference_roi.height / 2;
      }

      // Report result
      {
        PatchResult patch_result;
        result_roi.x = std::min(reference_image_.size().width  - result_roi.width - 1, std::max(0, result_roi.x));
        result_roi.y = std::min(reference_image_.size().height - result_roi.height - 1, std::max(0, result_roi.y));
        patch_result.first = image(result_roi);

        cv::Mat out = images_[image_index].clone();
        cv::circle(out, cv::Point2f(result_roi.x + result_roi.width / 2, result_roi.y + result_roi.height / 2), 40, cv::Scalar(0, 240, 0), 2);
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
