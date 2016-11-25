#include <roytracker/trackers/mean_shift_tracker.h>
#include <opencv2/imgproc.hpp>

void Tracker::setTargetObject(const cv::Mat &image, cv::Rect selection)
{
  type = TYPE_GENERIC;
  currentIteration = 0;
  currentSubIteration = 0;
  lastError = 0.0;
  lastTime = 0.0;
  converged = false;

  trackingWindow = selection;
  originalWindow = selection;

  cv::Mat newTarget = image(selection);
  newTarget.copyTo(targetObject);
}

cv::Point2f Tracker::getObjectCenter()
{
  return cv::Point2f(trackingWindow.x + trackingWindow.width / 2, trackingWindow.y + trackingWindow.height / 2);
}

void Tracker::setTargetObjectSelection(cv::Rect selection)
{
  trackingWindow = selection;
  originalWindow = selection;
}

MeanShiftTracker::MeanShiftTracker() {
  //this->type = MEAN_SHIFT;
  frameSize = cv::Rect(-1, -1, 0, 0);
  trackingWindow = cv::Rect(-1, -1, 0, 0);
  originalWindow = cv::Rect(-1, -1, 0, 0);
  trackingWindowCenter  = cv::Point(-1, -1);

  H = cv::Point(-1, -1);
  scalingFactor = (colorBins - 1) / 256.0;
  bhattaCoefficient = 0;
  currentScale = 1.0;

  //Variables para el Seguimiento
  bhattaEpsilon = 0.0005;
  maxIterations = 12;

  //Variables para enableModelUpdate Modelo
  enableModelUpdate = false;
  modelUpdateFactor = 97.0/100.0;
  modelUpdateEpsilon = 0.57;

  highlightColor = cv::Scalar(0xff, 0xe4, 0x00);
  highlightLineWidth = 2;
}

void MeanShiftTracker::setTargetObject(const cv::Mat &Img, cv::Rect trackingWindow)
{
  Tracker::setTargetObject(Img, trackingWindow);

  frameSize = cv::Rect(0, 0, Img.cols, Img.rows);

  //Inicializar variables
  H = cv::Point((int)(trackingWindow.width / 2.0), (int)(trackingWindow.height / 2.0));
  trackingWindowCenter = cv::Point( (int)(trackingWindow.x + trackingWindow.width / 2.0) , (int)(trackingWindow.y + trackingWindow.height / 2.0));

  //Variables para Ajustar trackingWindow scalingFactor
  enableWindowScaling = false;
  scalingFactorTest = 0.03;
  maxWindowChange = 0.6;
  maxWindow = cv::Rect(trackingWindow.x, trackingWindow.y, (int)((1.0 + maxWindowChange) * ((double)trackingWindow.width)), (int)((1.0 + maxWindowChange) * ((double)trackingWindow.height)));
  minWindow = cv::Rect(trackingWindow.x, trackingWindow.y, (int)((1.0 - maxWindowChange) * ((double)trackingWindow.width)), (int)((1.0 - maxWindowChange) * ((double)trackingWindow.height)));

  for (int r = 0; r < colorBins; r++)
    for (int g = 0; g < colorBins; g++)
      for (int b = 0; b < colorBins; b++)
        qu[r][g][b] = 0;

  getQu(Img);
}

void MeanShiftTracker::highlightObject(cv::Mat &outputImage)
{
  cv::rectangle(outputImage, cv::Point(trackingWindow.x, trackingWindow.y), cv::Point(trackingWindow.x + trackingWindow.width, trackingWindow.y + trackingWindow.height), highlightColor, highlightLineWidth);
  int pointerWidth = trackingWindow.width / 4;
  int pointerHeight = trackingWindow.height / 4;
  cv::line(outputImage, cv::Point(trackingWindowCenter.x - pointerWidth / 2, trackingWindowCenter.y), cv::Point(trackingWindowCenter.x + pointerWidth / 2, trackingWindowCenter.y), highlightColor, 1);
  cv::line(outputImage, cv::Point(trackingWindowCenter.x, trackingWindowCenter.y - pointerHeight / 2), cv::Point(trackingWindowCenter.x, trackingWindowCenter.y + pointerHeight / 2), highlightColor, 1);

  //    if (TrackingCommon::enableFilteringMeanShift) {
  //        cv::Rect trackingWindowFiltered = calculateTrackingWindow(trackingWindowCenterFiltered);
  //        cv::rectangle(outputImage, cv::Point(trackingWindowFiltered.x, trackingWindowFiltered.y), cv::Point(trackingWindowFiltered.x + trackingWindow.width, trackingWindowFiltered.y + trackingWindow.height), cv::Scalar(0, 0, 255), highlightLineWidth);
  //    }
}

void MeanShiftTracker::getQu(const cv::Mat &Img)
{
  int x, y;
  double norm = 0,tempC = 0, epani = 0;
  int indR,indG,indB;
  double x1, x2, limitOnX, limitOnY;
  cv::Scalar Corg, Camb;

  for ( int r = 0; r < colorBins; r++ ) {
    for ( int g = 0; g < colorBins; g++ )
      for ( int b = 0; b < colorBins; b++ ) {
        qu[r][g][b] = 0.0;
      }
  }

  limitOnX = trackingWindow.x + trackingWindow.width;
  limitOnY = trackingWindow.y + trackingWindow.height;

  for (y = trackingWindow.y; y < limitOnY; y++)
  {
    for (x = trackingWindow.x; x < limitOnX; x++) {
      x1 = (double)(trackingWindowCenter.x - x)/(H.x);
      x2 = (double)(trackingWindowCenter.y - y)/(H.y);

      norm = sqrt (x1 * x1 + x2 * x2);
      epani = norm < 1 ? (2.0 / 3.1416 )*(1 - norm * norm) : 0;
      tempC = tempC + epani;

      Corg = cv::Scalar(Img.at<cv::Vec3b>(y, x)[2], Img.at<Vec3b>(y, x)[1], Img.at<Vec3b>(y, x)[0]);

      indR = (int)(Corg.val[0] * scalingFactor);
      indG = (int)(Corg.val[1] * scalingFactor);
      indB = (int)(Corg.val[2] * scalingFactor);

      qu[indR][indG][indB] += epani;
    }
  }

  for ( int r = 0; r < colorBins; r++ ) {
    for ( int g = 0; g < colorBins; g++ )
      for ( int b = 0; b < colorBins; b++ ) {
        qu[r][g][b] = qu[r][g][b] / tempC;
      }
  }
}

void MeanShiftTracker::trackObject(const cv::Mat &image) {
  double temp = 0;
  double pu_actual[colorBins][colorBins][colorBins];
  double pu_nuevo[colorBins][colorBins][colorBins];
  cv::Point newCenter = cv::Point(-1, -1);
  cv::Point currentCenter = cv::Point(-1, -1);

  currentCenter = trackingWindowCenter;
  iteration = 1;

  do
  {
    getPu(image, pu_actual, currentCenter);
    newCenter = MeanShift(image, pu_actual, currentCenter);
    getPu(image, pu_nuevo, newCenter);

    temp = getBhattaCoefficient(qu, pu_nuevo) - getBhattaCoefficient(qu, pu_actual) ;
    if (temp < bhattaEpsilon || (iteration == maxIterations))
    {
      if (temp < 0)
        newCenter = currentCenter;

      bhattaCoefficient = getBhattaCoefficient(qu, pu_nuevo);
      break;
    }
    else
    {
      currentCenter = newCenter;
      iteration += 1;
    }
  }
  while (true);

  getPu(image, pu_nuevo, newCenter);
  double newPu = getBhattaCoefficient(qu, pu_nuevo);

  lastError = newPu;

  trackingWindowCenter = newCenter;

  if (enableModelUpdate)
    //if (getBhattaCoefficient(qu, pu_nuevo) < modelUpdateEpsilon)
    updateModel(image, newCenter);

  trackingWindow = calculateTrackingWindow(trackingWindowCenter);
}

cv::Point2f MeanShiftTracker::getObjectCenter()
{
  return trackingWindowCenter;
}

void MeanShiftTracker::getPu(const cv::Mat &Img, double pu[colorBins][colorBins][colorBins], cv::Point testPoint)
{
  int x, y;
  double norm = 0, tempC = 0, epani = 0;
  int indR, indG, indB;
  double x1, x2, limitOnX, limitOnY;
  cv::Scalar Corg, Camb;

  for ( int r = 0; r < colorBins; r++ )
    for ( int g = 0; g < colorBins; g++ )
      for ( int b = 0; b < colorBins; b++ )
        pu[r][g][b] = 0.0;

  // Obtiene la trackingWindow a partir del punto del prueba
  cv::Rect trackingWindowP = cv::Rect(-1, -1, 0, 0);
  trackingWindowP = calculateTrackingWindow(testPoint);

  limitOnX = trackingWindowP.x + trackingWindowP.width;
  limitOnY = trackingWindowP.y + trackingWindowP.height;

  for ( y = trackingWindowP.y; y < limitOnY; y++ ) {
    for ( x = trackingWindowP.x; x < limitOnX; x++ ) {
      x1 = (double)(testPoint.x - x)/(currentScale * H.x);
      x2 = (double)(testPoint.y - y)/(currentScale * H.y);

      norm = sqrt( x1 * x1 + x2 * x2 );
      epani = norm < 1 ? ( 2.0 / 3.1416 ) * ( 1 - norm * norm ) : 0;

      tempC = tempC + epani;

      Corg = cv::Scalar(Img.at<Vec3b>(y, x)[2], Img.at<Vec3b>(y, x)[1], Img.at<Vec3b>(y, x)[0]);

      indR = (int)(Corg.val[0] * scalingFactor);
      indG = (int)(Corg.val[1] * scalingFactor);
      indB = (int)(Corg.val[2] * scalingFactor);

      pu[indR][indG][indB] += epani;
    }
  }

  for ( int r = 0; r < colorBins; r++ ) {
    for ( int g = 0; g < colorBins; g++ )
      for ( int b = 0; b < colorBins; b++ ) {
        pu[r][g][b] = pu[r][g][b] / tempC;
      }
  }
}

cv::Point MeanShiftTracker::MeanShift(const cv::Mat &Img, double pu[colorBins][colorBins][colorBins], cv::Point testPoint) {
  double sampleDensity = 0, tempNumX = 0, tempNumY = 0;
  double sampleWeight = 0, kernelValue = 0, norm = 0;
  int x, y, indice = 0;
  cv::Point newPoint = cv::Point(-1, -1);
  int indR, indG, indB;
  double x1, x2, limitOnX, limitOnY;
  cv::Scalar Corg, Camb;

  cv::Rect trackingWindowP = cv::Rect(-1, -1, 0, 0);
  trackingWindowP = calculateTrackingWindow(testPoint);

  limitOnX = trackingWindowP.x + trackingWindowP.width;
  limitOnY = trackingWindowP.y + trackingWindowP.height;

  for( y = trackingWindowP.y; y < limitOnY; y++ ) {
    for( x = trackingWindowP.x; x < limitOnX; x++ ) {

      Corg = cv::Scalar(Img.at<Vec3b>(y, x)[2], Img.at<Vec3b>(y, x)[1], Img.at<Vec3b>(y, x)[0]);

      indR = (int)(Corg.val[0] * scalingFactor);
      indG = (int)(Corg.val[1] * scalingFactor);
      indB = (int)(Corg.val[2] * scalingFactor);

      if ( pu[indR][indG][indB] == 0 )
      {
        if ( qu[indR][indG][indB] == 0 )
          sampleWeight = 0.000001;
        else
          sampleWeight = 2;
      }
      else
        sampleWeight = (double)( (double)(qu[indR][indG][indB]) / (pu[indR][indG][indB]) ) ;

      x1 = (double)(testPoint.x - x)/(H.x);
      x2 = (double)(testPoint.y - y)/(H.y);

      norm = x1 * x1 + x2 * x2;
      kernelValue = norm <= 1 ? 1.0 : 0;

      tempNumX += x * sampleWeight * kernelValue;
      tempNumY += y * sampleWeight * kernelValue;
      sampleDensity += sampleWeight * kernelValue;
    }
  }

  if (sampleDensity != 0) {
    newPoint.x = (int)(tempNumX / sampleDensity) ;
    newPoint.y = (int)(tempNumY / sampleDensity) ;
  } else {
    return testPoint;
  }

  return newPoint;
}

double MeanShiftTracker::getBhattaCoefficient(double q[colorBins][colorBins][colorBins], double pu[colorBins][colorBins][colorBins])
{
  double dist = 0;
  for ( int r = 0; r < colorBins; r++ ) {
    for ( int g = 0; g < colorBins; g++)
      for ( int b = 0; b < colorBins; b++) {
        dist += sqrt( pu[r][g][b] * q[r][g][b] );
      }
  }
  return dist;
}

void MeanShiftTracker::updateModel(const cv::Mat &image, cv::Point p_trackingWindowCenter)
{
  double quUpdate[colorBins][colorBins][colorBins];
  double quWeighted[colorBins][colorBins][colorBins];
  double bhattaWeighted;
  double bhattaSource;

  // Update base image for displaying
  cv::Mat newModelImage = image(trackingWindow);
  newModelImage.copyTo(targetObject);

  // Get distribution at point
  getPu(image, quUpdate, p_trackingWindowCenter);

  for (int r = 0; r < colorBins; r++)
    for (int g = 0; g < colorBins; g++)
      for (int b = 0; b < colorBins; b++)
        quWeighted[r][g][b] = modelUpdateFactor * qu[r][g][b] + (1.0 - modelUpdateFactor) * quUpdate[r][g][b];

  bhattaWeighted = getBhattaCoefficient(quUpdate,quWeighted);
  bhattaSource = getBhattaCoefficient(quUpdate,originalQu);

  if ( bhattaWeighted >= bhattaSource )
  {
    for (int r = 0; r < colorBins; r++)
      for (int g = 0; g < colorBins; g++)
        for (int b = 0; b < colorBins; b++)
          qu[r][g][b] = quWeighted[r][g][b];
  }
  else
  {
    for (int r = 0; r < colorBins; r++)
      for (int g = 0; g < colorBins; g++)
        for (int b = 0; b < colorBins; b++)
          qu[r][g][b] = originalQu[r][g][b];
  }

  return;
}

cv::Point MeanShiftTracker::getWindowCenter()
{ 
  return cv::Point(trackingWindow.x + (int)(trackingWindow.width/2.0), trackingWindow.y + (int)(trackingWindow.height/2.0));
}

void MeanShiftTracker::updateTrackingWindowFromCenter(cv::Point windowCenter)
{
  trackingWindowCenter = windowCenter;
  trackingWindow = calculateTrackingWindow(trackingWindowCenter);
}

cv::Rect MeanShiftTracker::calculateTrackingWindow(cv::Point windowCenter)
{
  int difx, dify, incx, incy;
  cv::Rect testWindow = cv::Rect(-1, -1, 0, 0);

  testWindow.x = windowCenter.x - (int)(trackingWindow.width / 2);
  testWindow.y = windowCenter.y - (int)(trackingWindow.height / 2);
  testWindow.width = trackingWindow.width;
  testWindow.height = trackingWindow.height;

  dify = frameSize.height - trackingWindow.height;
  difx = frameSize.width  - trackingWindow.width;

  incx = testWindow.x + testWindow.width;
  incy = testWindow.y + testWindow.height;

  if ( testWindow.y < 0 )
    testWindow.y = 0;

  if ( testWindow.x < 0 )
    testWindow.x = 0;

  if (incx > frameSize.width)
    testWindow.x = difx;

  if (incy > frameSize.height)
    testWindow.y = dify;

  return testWindow;
}
