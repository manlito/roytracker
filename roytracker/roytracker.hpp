#ifndef ROYTRACKER_HPP
#define ROYTRACKER_HPP

#include <string>
#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace roytracker {

enum class KeypointDetector { AKAZE_KEYPOINTS };
enum class KeypointDescriptor { AKAZE_DESCRIPTOR };
enum class DescriptorMatcher { FLANN_MATCHER, BRUTEFORCE_MATCHER };
enum class ImageAlignmentMethod { CAM_SHIFT };

struct TrackingOptions {
  KeypointDetector detector;
  KeypointDescriptor descriptor;
  DescriptorMatcher matcher;
  ImageAlignmentMethod alignment_method;
  TrackingOptions(KeypointDetector detector = KeypointDetector::AKAZE_KEYPOINTS,
                  KeypointDescriptor descriptor = KeypointDescriptor::AKAZE_DESCRIPTOR,
                  DescriptorMatcher matcher = DescriptorMatcher::FLANN_MATCHER,
                  ImageAlignmentMethod alignment_method = ImageAlignmentMethod::CAM_SHIFT) :
    detector(detector), descriptor(descriptor), matcher(matcher),
    alignment_method(alignment_method) { }
};

struct TrackingObjectiveOptions {
  double x;
  double y;
  double width;
  double height;
  bool update_scale;
  bool update_orientation;
  TrackingObjectiveOptions(double x = 0.0,
                           double y = 0.0,
                           double width = 50,
                           double height = 50,
                           bool update_scale = true,
                           bool update_orientation = true) :
    x(x), y(y), width(width), height(height),
    update_orientation(update_orientation),
    update_scale(update_scale) { }
};

class TrackingObjective {
public:
  TrackingObjective(TrackingObjectiveOptions options) :
    options_(options) {}
  void SetInitialWindowSize(float width, float height);
  void SetScaleChange(bool enable);
  // Call SetImages to specify a custom subset of images to match
  void SetImages();
protected:
  // Typical options for an objective
  TrackingObjectiveOptions options_;

  // Image from where model is obtained
  size_t reference_image;
};

typedef std::map<size_t, std::pair<double, double>> RoyTrackerResult;
class RoyTracker {
public:
  RoyTracker(TrackingOptions options = TrackingOptions()) :
    options_(options) {}
  void AddImage(std::string filename) {
    images_.push_back(cv::imread(filename));
  }

  // Add tracking targets
  void AddTrackingObjective(const TrackingObjective &tracking_objective) {
    tracking_objectives.push_back(tracking_objective);
  }

  virtual void Track();

  // Utility functions
  void SetKeypointDetector(KeypointDetector detector);
  void SetKeypointDescriptor(KeypointDescriptor descriptor);
  void SetDescriptorMatcher(DescriptorMatcher matcher);
  void SetImageAlignmentMethod(ImageAlignmentMethod alignment_method);
protected:
  std::vector<TrackingObjective> tracking_objectives;
  std::vector<cv::Mat> images_;
  TrackingOptions options_;
};


}

#endif // ROYTRACKER_HPP
