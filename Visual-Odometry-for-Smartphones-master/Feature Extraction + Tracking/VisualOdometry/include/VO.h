#ifndef __VO__H__
#define __VO__H__

#include "iostream"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <vector>
#include <string.h>
#include <boost/lambda/bind.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include "map"
#include <fstream>



namespace VO{

    class featureOperations{
        public:
            //! Constructor for capturing from the camera
            featureOperations(cv::Mat cameraMatrix,bool drawMatches,bool enableHomography);

            //! Constructor for processing from a directory
            featureOperations(std::string imgFolderPath, cv::Mat cameraMatrix, bool drawMatches,bool enableHomography);

            //! Function for extracting features from an image
            std::vector<cv::Point2f> detectFeatures(cv::Mat img);

            //! Function to track features in consecutive images
            bool trackFeatures(cv::Mat prevImg,cv::Mat currentImg,std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, std::vector<uchar>& status);

            //! Plot Trajectory
            bool plotTrajectory(cv::Mat trajectory, cv::Mat translation, std::string windowName);

            //! For homography
            cv::Mat computeHomography(cv::Mat firstImage, cv::Mat secondImage);

            //! For homography from keypoints
            cv::Mat computeHomographyFromKeypoints(cv::Mat firstImage, cv::Mat secondImage, std::vector<cv::Point2f> FASTpoints1, std::vector<cv::Point2f> FASTpoints2);

        private:

            //! Camera Intrinsic Matrix
            cv::Mat m_intrinsicMatrix;

            //! Boolean Flag whether to display matches or not
            bool m_drawMatches;

    };

    class RobustMatcher {
    public:
      RobustMatcher() : ratio_(0.8f)
      {
        // ORB is the default feature
        detector_ = cv::ORB::create(1500);
        extractor_ = cv::ORB::create();

        // BruteForce matcher with Norm Hamming is the default matcher
        matcher_ = cv::makePtr<cv::BFMatcher>((int)cv::NORM_L2, false);

      }
      virtual ~RobustMatcher();

      // Set the feature detector
      void setFeatureDetector(const cv::Ptr<cv::FeatureDetector>& detect) {  detector_ = detect; }

      // Set the descriptor extractor
      void setDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor>& desc) { extractor_ = desc; }

      // Set the matcher
      void setDescriptorMatcher(const cv::Ptr<cv::DescriptorMatcher>& match) {  matcher_ = match; }

      // Compute the keypoints of an image
      void computeKeyPoints( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);

      // Compute the descriptors of an image given its keypoints
      void computeDescriptors( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

      // Set ratio parameter for the ratio test
      void setRatio( float rat) { ratio_ = rat; }

      // Clear matches for which NN ratio is > than threshold
      // return the number of removed points
      // (corresponding entries being cleared,
      // i.e. size will be 0)
      int ratioTest(std::vector<std::vector<cv::DMatch> > &matches);

      //RANSAC test
      std::vector<cv::Mat> ransacTest(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& outMatches, bool refineF);

      // Insert symmetrical matches in symMatches vector
      void symmetryTest( const std::vector<std::vector<cv::DMatch> >& matches1,
                         const std::vector<std::vector<cv::DMatch> >& matches2,
                         std::vector<cv::DMatch>& symMatches );

     // pointer to the matcher object
     cv::Ptr<cv::DescriptorMatcher> matcher_;

    private:
      // pointer to the feature point detector object
      cv::Ptr<cv::FeatureDetector> detector_;
      // pointer to the feature descriptor extractor object
      cv::Ptr<cv::DescriptorExtractor> extractor_;

      // max ratio between 1st and 2nd NN
      float ratio_;
    };

}
cv::Mat readCameraIntrinsic(std::string pathToIntrinsic);

#endif
