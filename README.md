# RoyTracker

It is a sort of tracker for Regions of Interest (Roy, for personality).

## How it Works

The implementation is simple and relies in [OpenCV](http://opencv.org/) as feature detection and matching backend. We have here however made something to make things easier for us, and hopefully for you:

- We are given an image coordinate, which we would like to track in other images
- Then, RoyTracker will search for good features around it
- Those matches are then matched using OpenCV in the indicated image indices
- Finally, we run an actual tracker (LK), to have the best possible alignment
- You will get a NCC score which you can use to accept or reject the matched patch.

Yes, we said patch. You will optionally say the initial patch size, but the tracker we are thinking will also look for scale. This method makes this serious assumptions:

- No serious changes in appearance (we have tested for aerial imagery)

My recommendation is to compile and use the ready to run demo. If it works in your case, awesome!

## Example

```c++
#include <map>
#include <roytracker/roytracker.hpp>
#include <opencv2/core.hpp>

using namespace roytracker;

void main() 
{
  RoyTracker tracker;
  tracker.LoadImage("image01.png"); // This will be image index 0
  tracker.LoadImage("image02.png"); // This will be image index 1
  tracker.LoadImage("image03.png"); // This will be image index 2
  tracker.SetKeypointDetector(ROYTRACKER_AKAZE_KEYPOINTS);
  tracker.SetDescriptor(ROYTRACKER_AKAZE_DESCRIPTOR);
  tracker.SetMatcher(ROYTRACKER_FLANN);
  tracker.SetImageAlignment(ROYTRACKER_LUCAS_KANADE);
  
  // You need to create a RoyTracker::TrackingObjective
  TrackingObjective objective(1200, 2000);
  objective.SetInitialWindowSize(100, 100);
  objective.SetScaleChange(true);
  objective.SetImages(ROYTRACKER_MATCH_ALL);
  
  // Store for results. This is a built-in typedef of std::map<size_t, cv::Point2d>>
  RoyTrackerResult correspondences;
  
  // Tell the coordinates to find and in which image to look for
  correspondences = tracker.Find(objective);
}

```

Or you need to look multiple points, use this, at it will avoid running matching more times than needed:

```c++
  // Store for results
  std::vector<RoyTrackerResult> correspondences;
  // Tell the coordinates to find
  std::vector<TrackingObjective> objectives;
  TrackingObjective objective_1(1200, 2000);
  objective_1.SetImages(std::vector<size_t> {0, 1});
  objective_1.SetScaleChange(true);
  TrackingObjective objective_2(2200, 3100);
  objective_2.SetImages(ROYTRACKER_MATCH_ALL);
  objective_2.SetScaleChange(false);

  objectives.push_back(objective_1);
  objectives.push_back(objective_2);
  correspondences = tracker.Find(objectives);
```

