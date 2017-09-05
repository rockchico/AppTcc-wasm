#include "VO.h"

    // recovering the pose and the essential matrix
    cv::Mat E, R, t, mask;
    cv::Mat homography;

void VO::featureOperations::release() { // Loading images from a camera
    E.release();
    R.release();
    t.release();
    mask.release();
    homography.release();

}    

VO::featureOperations::featureOperations(cv::Mat cameraMatrix, bool drawMatches,bool enableHomography){ // Loading images from a camera
    this->m_drawMatches=drawMatches;
    this->m_intrinsicMatrix=cameraMatrix;

}

VO::featureOperations::featureOperations(cv::Mat firstImage, cv::Mat& secondImage, cv::Mat cameraMatrix, bool drawMatches, bool enableHomography){ // Loading images from a directory

    // Initializing
    this->m_drawMatches=drawMatches;
    this->m_intrinsicMatrix=cameraMatrix;
    


    // First two images
    std::string currentImg,nextImg;


    if ( !firstImage.data || !secondImage.data ) {
      std::cout<< " --(!) Error reading images " << std::endl;
    }


    std::vector<uchar> status;
    std::vector<cv::Point2f> points1,points2;

    points1 = this->detectFeatures(firstImage);

    // Change these accordingly. Intrinsics

    double focal = 718.8560;
    cv::Point2d pp(607.1928, 185.2157);




    if(this->trackFeatures(firstImage,secondImage,points1,points2,status)){
        if(enableHomography){
        homography=this->computeHomographyFromKeypoints(firstImage,secondImage,points1,points2);
//            homography=this->computeHomography(firstImage,secondImage);
        }
        E = cv::findEssentialMat(points2, points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
        cv::recoverPose(E, points2, points1, R, t, focal, pp, mask);
    }


    cv::Mat prevImage = secondImage;
    cv::Mat currImage;
    std::vector<cv::Point2f> prevFeatures = points2;
    std::vector<cv::Point2f> currFeatures;

    double scale = 1;
    
            //cout << "Scale is " << scale << endl;
    
    if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {

            t = t + scale*(R*t);
            R = R*R;

    }

    




}

std::vector<cv::Point2f> VO::featureOperations::detectFeatures(cv::Mat img){
    cv::Mat distortCoeffs;
//    cv::Mat undistortedImg;

    cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
    // Undistort

//    cv::undistort(img,undistortedImg,this->m_intrinsicMatrix,distortCoeffs);

//    cv::Mat imgForOrb = undistortedImg.clone();

    // Detect features on this image
    std::vector<cv::Point2f> pointsFAST;
    std::vector<cv::KeyPoint> keypoints_FAST;

    // FAST Detector
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    cv::FAST(img,keypoints_FAST,fast_threshold,nonmaxSuppression);
    cv::KeyPoint::convert(keypoints_FAST,pointsFAST,std::vector<int>());
    assert(pointsFAST.size() > 0);
    return pointsFAST;

    // SHi Tomasi

//    std::vector<cv::Point2f> pointsShiTomasi;
//    double qualityLevel = 0.01;
//    double minDistance = 10;
//    int blockSize = 3;
//    bool useHarrisDetector = false;
//    double k = 0.04;
//    cv::goodFeaturesToTrack(img,pointsShiTomasi,23,qualityLevel,minDistance,cv::Mat(),blockSize,useHarrisDetector,k);
//    return pointsShiTomasi;
}

bool VO::featureOperations::trackFeatures(cv::Mat prevImg, cv::Mat currentImg, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, std::vector<uchar>& status){

    cv::Mat prevImg_gray,currentImg_gray;
    cv::cvtColor(prevImg,prevImg_gray,CV_RGBA2GRAY);
    cv::cvtColor(currentImg,currentImg_gray,CV_RGBA2GRAY);

    std::vector<float> err;
    cv::Size winSize=cv::Size(21,21);
    cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);


    cv::calcOpticalFlowPyrLK(prevImg_gray, currentImg_gray, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for( int i=0; i<status.size(); i++)
       {  cv::Point2f pt = points2.at(i- indexCorrection);
          if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
                if((pt.x<0)||(pt.y<0))	{
                  status.at(i) = 0;
                }
                points1.erase (points1.begin() + (i - indexCorrection));
                points2.erase (points2.begin() + (i - indexCorrection));
                indexCorrection++;
          }

       }

//    std::cout<<"Size of Points 1 : "<<points1.size()<<" Points 2 : "<<points2.size()<<std::endl;
    if(points1.size()<=5||points2.size()<=5){
        std::cout<<"Previous Features : \n"<<points1<<std::endl;
        std::cout<<"Current Features : \n"<<points2<<std::endl;
    }
    return true;
}


// bool VO::featureOperations::plotTrajectory(cv::Mat trajectory,cv::Mat translation,std::string windowName){

//     cv::namedWindow(windowName,CV_WINDOW_AUTOSIZE);

//     int x = int(translation.at<double>(0)) + 300;
//     int y = int(translation.at<double>(2)) + 100;
//     cv::circle(trajectory, cv::Point(x, y) ,1, CV_RGB(255,0,0), 2);

//     char text[100];
//     int fontFace = CV_FONT_HERSHEY_PLAIN;
//     double fontScale = 1;
//     int thickness = 1;
//     cv::Point textOrg(10, 50);

//     cv::rectangle( trajectory, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
//     sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", translation.at<double>(0), translation.at<double>(1), translation.at<double>(2));
//     cv::putText(trajectory, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
//     cv::imshow(windowName,trajectory);
//     cv::waitKey(30);

//     return true;

// }

cv::Mat readCameraIntrinsic(){
   float cameraIntrinsicMatrix[3][3];
   cv::Mat cameraMatrix;
   

    //514.3332    0           317.0206
    //0           511.1454    243.0076
    //0           0           1

    cameraIntrinsicMatrix[0][0] = 514.3332;
    cameraIntrinsicMatrix[0][1] = 0;
    cameraIntrinsicMatrix[0][2] = 317.0206;
    
    cameraIntrinsicMatrix[1][0] = 0;
    cameraIntrinsicMatrix[1][1] = 511.1454;
    cameraIntrinsicMatrix[1][2] = 243.0076;

    cameraIntrinsicMatrix[2][0] = 0;
    cameraIntrinsicMatrix[2][1] = 0;
    cameraIntrinsicMatrix[2][2] = 1;



   cameraMatrix=cv::Mat(3,3,CV_32FC1,&cameraIntrinsicMatrix);
   std::cout<<"\n ******Camera Intrinsic Matrix********** \n"<<std::endl;
//   std::cout<<cameraMatrix.at<float>(1,1)<<std::endl;


   return cameraMatrix;
}

cv::Mat VO::featureOperations::computeHomography(cv::Mat firstImage, cv::Mat secondImage){
    // Checking with Robust Matcher
            std::vector<cv::KeyPoint> FASTKeypoints1,FASTKeypoints2;
            cv::Mat orbDescriptors1,orbDescriptors2;
            cv::Mat orbDescriptors1_8U,orbDescriptors2_8U;


            VO::RobustMatcher robustMatcher;


            int fast_threshold = 20;
            bool nonmaxSuppression = true;

            cv::FAST(firstImage,FASTKeypoints1,fast_threshold,nonmaxSuppression);
            cv::FAST(secondImage,FASTKeypoints2,fast_threshold,nonmaxSuppression);

            robustMatcher.computeDescriptors(firstImage,FASTKeypoints1,orbDescriptors1);
            robustMatcher.computeDescriptors(secondImage,FASTKeypoints2,orbDescriptors2);

            if ( orbDescriptors1.empty() )
               cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);
            if ( orbDescriptors2.empty() )
               cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);

            if(orbDescriptors1.type()!=CV_32F) {
                orbDescriptors1.convertTo(orbDescriptors1, CV_32F);
                orbDescriptors1.convertTo(orbDescriptors1_8U, CV_8U);

            }

            if(orbDescriptors2.type()!=CV_32F) {
                orbDescriptors2.convertTo(orbDescriptors2, CV_32F);
                orbDescriptors2.convertTo(orbDescriptors2_8U, CV_8U);
            }

            std::vector<std::vector<cv::DMatch> > matches1;
            robustMatcher.matcher_->knnMatch(orbDescriptors1,orbDescriptors2,matches1,2);

            std::vector<std::vector<cv::DMatch> > matches2;
            robustMatcher.matcher_->knnMatch(orbDescriptors2,orbDescriptors1,matches2,2);


            int removed = robustMatcher.ratioTest(matches1);
            removed = robustMatcher.ratioTest(matches2);

            std::vector<cv::DMatch> symMatches;
            robustMatcher.symmetryTest(matches1,matches2,symMatches);

            // Matching descriptor vectors using FLANN matcher
            cv::FlannBasedMatcher flannmatcher;
            std::vector< cv::DMatch > matches;
            flannmatcher.match( orbDescriptors1, orbDescriptors2, matches );


            double max_dist = 0; double min_dist = 2000;

            //-- Quick calculation of max and min distances between keypoints
            for( int i = 0; i < orbDescriptors1.rows; i++ )
            { double dist = matches[i].distance;
              if( dist < min_dist ) min_dist = dist;
              if( dist > max_dist ) max_dist = dist;
            }

//            std::cout<<"Min Distance : "<<min_dist<<" Max Distance : "<<max_dist<<std::endl;

            std::vector< cv::DMatch > good_matches;

            for( int i = 0; i < orbDescriptors1.rows; i++ )
              { if( matches[i].distance <= cv::max(5*min_dist, 200.0) )
                {
                    good_matches.push_back( matches[i]);
                }
              }

            //-- Draw only "good" matches
            cv::Mat img_matches;


            //-- Localize the object
            std::vector<cv::Point2f> img1Keypoints;
            std::vector<cv::Point2f> img2Keypoints;

            for( int i = 0; i < good_matches.size(); i++ )
              {
                //-- Get the keypoints from the good matches
                img1Keypoints.push_back( FASTKeypoints1[ good_matches[i].queryIdx ].pt );
                img2Keypoints.push_back( FASTKeypoints2[ good_matches[i].trainIdx ].pt );
              }

            cv::Mat H = cv::findHomography(img1Keypoints,img2Keypoints,CV_RANSAC);


            if(this->m_drawMatches){

                cv::drawMatches( firstImage, FASTKeypoints1, secondImage, FASTKeypoints2,good_matches, img_matches,
                                 cv::Scalar::all(-1), cv::Scalar::all(-1),std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

                //-- Get the corners from the image_1 ( the object to be "detected" )
                 std::vector<cv::Point2f> img1_corners(4);
                 img1_corners[0] = cv::Point(0,0); img1_corners[1] = cv::Point( firstImage.cols, 0 );
                 img1_corners[2] = cv::Point( firstImage.cols, firstImage.rows ); img1_corners[3] = cv::Point( 0, firstImage.rows );

                 std::vector<cv::Point2f> img2_corners(4);

                 if(!H.empty()){
                    cv::perspectiveTransform( img1_corners, img2_corners, H);

                     //-- Draw lines between the corners (the mapped object in the scene - image_2 )
                     cv::line( img_matches, img2_corners[0] + cv::Point2f( firstImage.cols, 0), img2_corners[1] + cv::Point2f( firstImage.cols, 0), cv::Scalar(0, 255, 0), 4 );
                     cv::line( img_matches, img2_corners[1] + cv::Point2f( firstImage.cols, 0), img2_corners[2] + cv::Point2f( firstImage.cols, 0), cv::Scalar( 0, 255, 0), 4 );
                     cv::line( img_matches, img2_corners[2] + cv::Point2f( firstImage.cols, 0), img2_corners[3] + cv::Point2f( firstImage.cols, 0), cv::Scalar( 0, 255, 0), 4 );
                     cv::line( img_matches, img2_corners[3] + cv::Point2f( firstImage.cols, 0), img2_corners[0] + cv::Point2f( firstImage.cols, 0), cv::Scalar( 0, 255, 0), 4 );
                 }
                 else{
                     std::cout<<"No Homography"<<std::endl;
                 }


    //            std::cout<<H<<std::endl;
            }


            if(!H.empty()){
                std::cout<<"Homography Computed"<<std::endl;
            }
            return H;

}

cv::Mat VO::featureOperations::computeHomographyFromKeypoints(cv::Mat firstImage, cv::Mat secondImage, std::vector<cv::Point2f> FASTpoints1, std::vector<cv::Point2f> FASTpoints2){
    // Checking with Robust Matcher
            cv::Mat orbDescriptors1,orbDescriptors2;
            std::vector<cv::KeyPoint> FASTKeypoints1,FASTKeypoints2;

            cv::KeyPoint::convert(FASTpoints1,FASTKeypoints1);
            cv::KeyPoint::convert(FASTpoints2,FASTKeypoints2);

            VO::RobustMatcher robustMatcher;

            robustMatcher.computeDescriptors(firstImage,FASTKeypoints1,orbDescriptors1);
            robustMatcher.computeDescriptors(secondImage,FASTKeypoints2,orbDescriptors2);

            if(orbDescriptors1.type()!=CV_32F) {
                orbDescriptors1.convertTo(orbDescriptors1, CV_32F);
            }

            if(orbDescriptors2.type()!=CV_32F) {
                orbDescriptors2.convertTo(orbDescriptors2, CV_32F);
            }


            if ( orbDescriptors1.empty() )
               cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);
            if ( orbDescriptors2.empty() )
               cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);

            std::vector<std::vector<cv::DMatch> > matches1;
            robustMatcher.matcher_->knnMatch(orbDescriptors1,orbDescriptors2,matches1,2);

            std::vector<std::vector<cv::DMatch> > matches2;
            robustMatcher.matcher_->knnMatch(orbDescriptors2,orbDescriptors1,matches2,2);


            int removed = robustMatcher.ratioTest(matches1);
            removed = robustMatcher.ratioTest(matches2);

            std::vector<cv::DMatch> symMatches;
            robustMatcher.symmetryTest(matches1,matches2,symMatches);

            // Matching descriptor vectors using FLANN matcher
            cv::FlannBasedMatcher flannmatcher;
            std::vector< cv::DMatch > matches;
            flannmatcher.match( orbDescriptors1, orbDescriptors2, matches );



            double max_dist = 0; double min_dist = 2000;

            //-- Quick calculation of max and min distances between keypoints
            for( int i = 0; i < orbDescriptors1.rows; i++ )
            { double dist = matches[i].distance;
              if( dist < min_dist ) min_dist = dist;
              if( dist > max_dist ) max_dist = dist;
            }

//            std::cout<<"Min Distance : "<<min_dist<<" Max Distance : "<<max_dist<<std::endl;

            std::vector< cv::DMatch > good_matches;

            for( int i = 0; i < orbDescriptors1.rows; i++ )
              { if( matches[i].distance <= cv::max(5*min_dist, 200.0) )
                {
                    good_matches.push_back( matches[i]);
                }
              }

            //-- Draw only "good" matches
            cv::Mat img_matches;


            //-- Localize the object
            std::vector<cv::Point2f> img1Keypoints;
            std::vector<cv::Point2f> img2Keypoints;

            for( int i = 0; i < good_matches.size(); i++ )
              {
                //-- Get the keypoints from the good matches
                img1Keypoints.push_back( FASTKeypoints1[ good_matches[i].queryIdx ].pt );
                img2Keypoints.push_back( FASTKeypoints2[ good_matches[i].trainIdx ].pt );
              }

            cv::Mat H = cv::findHomography(img1Keypoints,img2Keypoints,CV_RANSAC);

            //-- Get the corners from the image_1 ( the object to be "detected" )
             std::vector<cv::Point2f> img1_corners(4);
             img1_corners[0] = cv::Point(0,0); img1_corners[1] = cv::Point( firstImage.cols, 0 );
             img1_corners[2] = cv::Point( firstImage.cols, firstImage.rows ); img1_corners[3] = cv::Point( 0, firstImage.rows );

             std::vector<cv::Point2f> img2_corners(4);

             if(this->m_drawMatches){
                 cv::drawMatches( firstImage, FASTKeypoints1, secondImage, FASTKeypoints2,good_matches, img_matches,
                                 cv::Scalar::all(-1), cv::Scalar::all(-1),std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

                 if(!H.empty()){
                    cv::perspectiveTransform( img1_corners, img2_corners, H);

                     //-- Draw lines between the corners (the mapped object in the scene - image_2 )
                     cv::line( img_matches, img2_corners[0] + cv::Point2f( firstImage.cols, 0), img2_corners[1] + cv::Point2f( firstImage.cols, 0), cv::Scalar(0, 255, 0), 4 );
                     cv::line( img_matches, img2_corners[1] + cv::Point2f( firstImage.cols, 0), img2_corners[2] + cv::Point2f( firstImage.cols, 0), cv::Scalar( 0, 255, 0), 4 );
                     cv::line( img_matches, img2_corners[2] + cv::Point2f( firstImage.cols, 0), img2_corners[3] + cv::Point2f( firstImage.cols, 0), cv::Scalar( 0, 255, 0), 4 );
                     cv::line( img_matches, img2_corners[3] + cv::Point2f( firstImage.cols, 0), img2_corners[0] + cv::Point2f( firstImage.cols, 0), cv::Scalar( 0, 255, 0), 4 );
                 }
                 else{
                     std::cout<<"No Homography"<<std::endl;
                 }

                
             }

             if(!H.empty()){
                 std::cout<<"Homography Computed"<<std::endl;
             }
             return H;

}

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