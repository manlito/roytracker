#ifndef MEAN_SHIFT_H
#define MEAN_SHIFT_H

#include <opencv2/core/core.hpp>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

class Tracker
{
public:
  int currentIteration;
  int currentSubIteration;
  cv::Rect trackingWindow;
  cv::Rect originalWindow;
  cv::Point2f trackingWindowCenter;
  cv::Point2f trackingWindowCenterFiltered;
  cv::Mat targetObject;
  cv::Scalar highlightColor;
  int highlightLineWidth;
  double lastError;
  double lastAverageError;
  cv::Mat lastParameters;
  float lastTime;
  // Used to indicate if the algorithm converged on last iteration
  bool converged;
  float lastProjectionError;
  enum TrackerTypes {TYPE_GENERIC, TYPE_MEAN_SHIFT, TYPE_ESM, TYPE_ESM_PYRAMIDAL, TYPE_CUBOID, TYPE_CUBOID_SINGLE_FACE} type;

  virtual void trackObject(const cv::Mat &newImage) = 0;
  virtual void highlightObject(cv::Mat &targetImage) = 0;
  virtual cv::Point2f getObjectCenter();
  virtual void setTargetObject(const cv::Mat &image, cv::Rect selection);
  virtual void setTargetObjectSelection(cv::Rect selection);
};

class MeanShiftTracker : public Tracker
{
public:
  static const int colorBins = 16;

  cv::Rect frameSize;

  cv::Point H;
  int iteration;
  double scalingFactor;
  double bhattaCoefficient;

  double qu[colorBins][colorBins][colorBins];
  double originalQu[colorBins][colorBins][colorBins];

  double bhattaEpsilon;
  int maxIterations;

  // Scale adaptation test
  bool enableWindowScaling;
  double scalingFactorTest;
  double maxWindowChange;

  // Manuel > Current scale for tracking window
  double currentScale;
  cv::Rect maxWindow;
  cv::Rect minWindow;

  // Model Update
  bool enableModelUpdate;
  double modelUpdateFactor;
  double modelUpdateEpsilon;

  // Display variables
  CvScalar frameColor;

  MeanShiftTracker();
  void setTargetObject(const cv::Mat &Img, cv::Rect trackingWindow);
  void trackObject(const cv::Mat &newImage);
  void highlightObject(cv::Mat &outputImage);
  cv::Point2f getObjectCenter();

  void getQu(const cv::Mat &Img);
  void getPu(const cv::Mat &Img, double pu[colorBins][colorBins][colorBins], cv::Point testPoint);
  cv::Point MeanShift(const cv::Mat &Img, double pu[colorBins][colorBins][colorBins], cv::Point testPoint);
  double getBhattaCoefficient(double q[colorBins][colorBins][colorBins], double pu[colorBins][colorBins][colorBins]);
  cv::Point getWindowCenter();
  void updateTrackingWindowFromCenter(cv::Point windowCenter);
  cv::Rect calculateTrackingWindow(cv::Point windowCenter);
  void updateModel(const cv::Mat &bufferedImage, cv::Point p_trackingWindowCenter);

};

#endif
