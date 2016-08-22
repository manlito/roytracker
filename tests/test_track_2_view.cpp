#include <functional>
#include <vector>
#include "gtest/gtest.h"
#include <roytracker/roytracker.hpp>

using namespace roytracker;

TEST(Tracker, TwoView) {
  // Set up tracking options
  TrackingObjectiveOptions tracking_options;
  tracking_options.x = 1680;
  tracking_options.y = 1192;
  tracking_options.width = 90;
  tracking_options.height = 90;
  tracking_options.update_scale = false;
  tracking_options.update_orientation = true;

  // Set up a tracking objective
  TrackingObjective tracking_objective(tracking_options);

  // Set up main tracker
  RoyTracker tracker;
  tracker.AddTrackingObjective(tracking_objective);

  // Add images to track
  tracker.SetReferenceImage(ROYTRACKER_SAMPLE_IMAGE_00);
  tracker.AddImage(ROYTRACKER_SAMPLE_IMAGE_01);

  std::vector<TrackingResult> results;
  tracker.Track(results);

  // For debugging
  std::string reference_patch_filename =
      std::string(ROYTRACKER_SAMPLE_OUTPUT_FOLDER) + "/reference_patch.jpg";
  cv::imwrite(reference_patch_filename, results[0].reference_patch);
}
