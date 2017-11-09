#include "src/VO.h"



VO::RobustMatcher::~RobustMatcher()
{
  // TODO Auto-generated destructor stub
}

void VO::RobustMatcher::computeKeyPoints( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints)
{
  detector_->detect(image, keypoints);
}

void VO::RobustMatcher::computeDescriptors( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
  extractor_->compute(image, keypoints, descriptors);
}

int VO::RobustMatcher::ratioTest(std::vector<std::vector<cv::DMatch> > &matches)
{
  int removed = 0;
  // for all matches
  for ( std::vector<std::vector<cv::DMatch> >::iterator
        matchIterator= matches.begin(); matchIterator!= matches.end(); ++matchIterator)
  {
    // if 2 NN has been identified
    if (matchIterator->size() > 1)
    {
      // check distance ratio
      if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > ratio_)
      {
        matchIterator->clear(); // remove match
        removed++;
      }
    }
    else
    { // does not have 2 neighbours
      matchIterator->clear(); // remove match
      removed++;
    }
  }
  return removed;
}

void VO::RobustMatcher::symmetryTest( const std::vector<std::vector<cv::DMatch> >& matches1,
                     const std::vector<std::vector<cv::DMatch> >& matches2,
                     std::vector<cv::DMatch>& symMatches )
{

  // for all matches image 1 -> image 2
   for (std::vector<std::vector<cv::DMatch> >::const_iterator
       matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1)
   {

      // ignore deleted matches
      if (matchIterator1->empty() || matchIterator1->size() < 2)
         continue;

      // for all matches image 2 -> image 1
      for (std::vector<std::vector<cv::DMatch> >::const_iterator
          matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2)
      {
        // ignore deleted matches
        if (matchIterator2->empty() || matchIterator2->size() < 2)
           continue;

        // Match symmetry test
        if ((*matchIterator1)[0].queryIdx ==
            (*matchIterator2)[0].trainIdx &&
            (*matchIterator2)[0].queryIdx ==
            (*matchIterator1)[0].trainIdx)
        {
            // add symmetrical match
            symMatches.push_back(
              cv::DMatch((*matchIterator1)[0].queryIdx,
                         (*matchIterator1)[0].trainIdx,
                         (*matchIterator1)[0].distance));
            break; // next match in image 1 -> image 2
        }
      }
   }

}

std::vector<cv::Mat> VO::RobustMatcher::ransacTest(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2,
                                      std::vector<cv::DMatch> &outMatches, bool refineF){

      std::cout<<"Ransac Test"<<std::endl;
      // Convert keypoints into Point2f
      std::vector<cv::Point2f> points1, points2;
      cv::Mat essential;
      cv::Mat R,t,mask;

      for (std::vector<cv::DMatch>::const_iterator it= matches.begin();it!= matches.end(); ++it) {
          // Get the position of left keypoints
          float x= keypoints1[it->queryIdx].pt.x;
          float y= keypoints1[it->queryIdx].pt.y;
          points1.push_back(cv::Point2f(x,y));
          // Get the position of right keypoints
          x= keypoints2[it->trainIdx].pt.x;
          y= keypoints2[it->trainIdx].pt.y;
          points2.push_back(cv::Point2f(x,y));
       }

      // Compute F matrix using RANSAC
      std::vector<uchar> inliers(points1.size(),0);


      if (points1.size()>0 && points2.size()>0){

         essential = cv::findEssentialMat(points1,points2,1.0,cv::Point2d(0,0),CV_RANSAC,0.999,1.0,cv::noArray());
         cv::recoverPose(essential, points2, points1, R, t, 1.0,cv::Point2d(0,0), mask);


         // extract the surviving (inliers) matches
         std::vector<uchar>::const_iterator itIn= inliers.begin();
         std::vector<cv::DMatch>::const_iterator itM= matches.begin();

         // for all matches
         for ( ;itIn!= inliers.end(); ++itIn, ++itM) {
            if (*itIn) { // it is a valid match
                outMatches.push_back(*itM);
             }
          }

          if (refineF) {
             // Convert keypoints into Point2f
             // for final F computation
             points1.clear();
             points2.clear();

             for (std::vector<cv::DMatch>::const_iterator it= outMatches.begin();it!= outMatches.end(); ++it) {
                 // Get the position of left keypoints
                 float x= keypoints1[it->queryIdx].pt.x;
                 float y= keypoints1[it->queryIdx].pt.y;
                 points1.push_back(cv::Point2f(x,y));
                 // Get the position of right keypoints
                 x= keypoints2[it->trainIdx].pt.x;
                 y= keypoints2[it->trainIdx].pt.y;
                 points2.push_back(cv::Point2f(x,y));
             }

             // Compute 8-point F from all accepted matches
             if (points1.size()>0 && points2.size()>0){
                essential = cv::findEssentialMat(points1,points2,1.0,cv::Point2d(0,0),CV_RANSAC,0.999,1.0,cv::noArray());
                cv::recoverPose(essential, points2, points1, R, t, 1.0,cv::Point2d(0,0), mask);
             }
          }
       }

       std::vector<cv::Mat> transform;
       transform.push_back(t);
       transform.push_back(R);
       return transform;
}
