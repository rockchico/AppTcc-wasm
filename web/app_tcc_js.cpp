/*
    homografia
    http://www.docs.opencv.org/2.4.13/doc/tutorials/features2d/feature_homography/feature_homography.html

    mono-vo
    https://github.com/avisingh599/mono-vo
*/




#include <stdio.h>
#include <iostream>
#include <string>

#include <emscripten.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
//#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"


#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include "VO.h"



using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


// global data

cv::Mat frame_base(240, 320, CV_8UC3, Scalar(0,0,0));

cv::Mat mat_white(240, 320, CV_8UC3, Scalar(255,255,255));
//cv::Mat frame_white(240, 320, CV_8UC3, Scalar(255,255,255));

cv::Mat firstImage(240, 320, CV_8UC3, Scalar(0,0,0));
cv::Mat secondImage(240, 320, CV_8UC3, Scalar(0,0,0));

std::vector<cv::KeyPoint> FASTKeypoints1,FASTKeypoints2;
cv::Mat orbDescriptors1,orbDescriptors2;
cv::Mat orbDescriptors1_8U,orbDescriptors2_8U;

vector<Point2f> points1;
vector<Point2f> points2;

extern "C" 
{

    void filledCircle( Mat& img, Point center, int radius, cv::Scalar cor ) {
        int thickness = -1;
        int lineType = 8;

        circle( img,
                center,
                radius,
                cor,
                thickness,
                lineType );
    }

    void featureDetection(Mat& img, vector<Point2f>& points)	{   //uses FAST as of now, modify parameters as necessary
        vector<KeyPoint> keypoints;


        // original = 20
        // quanto o maior o fast_threshold menos pontos capturados
        int fast_threshold = 70;
        bool nonmaxSuppression = true;
        FAST(img, keypoints, fast_threshold, nonmaxSuppression);

        // filtra os melhores 100 pontos
        // http://answers.opencv.org/question/12316/set-a-threshold-on-fast-feature-detection/
        cv::KeyPointsFilter::retainBest(keypoints, 100);


        KeyPoint::convert(keypoints, points, vector<int>());

        //drawKeypoints(img, keypoints, img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

        std::cout<<"Num features: "<<points.size()<<std::endl;

    }

    void featureDetection2(Mat& img, vector<Point2f>& points)	{   //uses FAST as of now, modify parameters as necessary
            // SHi Tomasi
       double qualityLevel = 0.01;
       double minDistance = 10;
       int blockSize = 3;
       bool useHarrisDetector = false;
       double k = 0.04;
       cv::goodFeaturesToTrack(img,points,23,qualityLevel,minDistance,cv::Mat(),blockSize,useHarrisDetector,k);

       std::cout<<"Num features: "<<points.size()<<std::endl;

    }






    void featureTracking(Mat& img_1, Mat& img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{ 
        
        //this function automatically gets rid of points for which tracking fails
    
        vector<float> err;					
        Size winSize=Size(21,21);																								
        TermCriteria termcrit = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    
        calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
    
        //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
        int indexCorrection = 0;
        for( int i=0; i<status.size(); i++) {  
            Point2f pt = points2.at(i- indexCorrection);
                if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
                    if((pt.x<0)||(pt.y<0))	{
                        status.at(i) = 0;
                    }
                    points1.erase (points1.begin() + (i - indexCorrection));
                    points2.erase (points2.begin() + (i - indexCorrection));
                    indexCorrection++;
                }
    
        }

        std::cout<<"Size of Points 1 : "<<points1.size()<<" Points 2 : "<<points2.size()<<std::endl;


        //cv::drawMatches(img_1, points1, img_2, points2, status, img_2);
        // cv::drawMatches(img_1, points1, img_2, points2, status, img_2, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);
        // cv::drawMatches( firstImage, FASTKeypoints1, secondImage, FASTKeypoints2,good_matches, img_matches,
        //     cv::Scalar::all(-1), cv::Scalar::all(-1),std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


    
    }

//     cv::Mat computeHomographyFromKeypoints(cv::Mat firstImage, cv::Mat& secondImage, std::vector<cv::Point2f> FASTpoints1, std::vector<cv::Point2f> FASTpoints2, bool drawMatches){
//         // Checking with Robust Matcher
//         cv::Mat orbDescriptors1,orbDescriptors2;
//         std::vector<cv::KeyPoint> FASTKeypoints1,FASTKeypoints2;

//         cv::KeyPoint::convert(FASTpoints1,FASTKeypoints1);
//         cv::KeyPoint::convert(FASTpoints2,FASTKeypoints2);

//         VO::RobustMatcher robustMatcher;

//         robustMatcher.computeDescriptors(firstImage,FASTKeypoints1,orbDescriptors1);
//         robustMatcher.computeDescriptors(secondImage,FASTKeypoints2,orbDescriptors2);

//         if(orbDescriptors1.type()!=CV_32F) {
//             orbDescriptors1.convertTo(orbDescriptors1, CV_32F);
//         }

//         if(orbDescriptors2.type()!=CV_32F) {
//             orbDescriptors2.convertTo(orbDescriptors2, CV_32F);
//         }


//         if ( orbDescriptors1.empty() )
//             cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);
//         if ( orbDescriptors2.empty() )
//             cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);

//         std::vector<std::vector<cv::DMatch> > matches1;
//         robustMatcher.matcher_->knnMatch(orbDescriptors1,orbDescriptors2,matches1,2);

//         std::vector<std::vector<cv::DMatch> > matches2;
//         robustMatcher.matcher_->knnMatch(orbDescriptors2,orbDescriptors1,matches2,2);


//         int removed = robustMatcher.ratioTest(matches1);
//         removed = robustMatcher.ratioTest(matches2);

//         std::vector<cv::DMatch> symMatches;
//         robustMatcher.symmetryTest(matches1,matches2,symMatches);

//         // Matching descriptor vectors using FLANN matcher
//         cv::FlannBasedMatcher flannmatcher;
//         std::vector< cv::DMatch > matches;
//         flannmatcher.match( orbDescriptors1, orbDescriptors2, matches );



//         double max_dist = 0; double min_dist = 2000;

//         //-- Quick calculation of max and min distances between keypoints
//         for( int i = 0; i < orbDescriptors1.rows; i++ )
//         { double dist = matches[i].distance;
//             if( dist < min_dist ) min_dist = dist;
//             if( dist > max_dist ) max_dist = dist;
//         }

// //            std::cout<<"Min Distance : "<<min_dist<<" Max Distance : "<<max_dist<<std::endl;

//         std::vector< cv::DMatch > good_matches;

//         for( int i = 0; i < orbDescriptors1.rows; i++ ) { 
//             if( matches[i].distance <= cv::max(5*min_dist, 200.0) ) {
//                 good_matches.push_back( matches[i]);
//             }
//         }

//         //-- Draw only "good" matches
//         cv::Mat img_matches;


//         //-- Localize the object
//         std::vector<cv::Point2f> img1Keypoints;
//         std::vector<cv::Point2f> img2Keypoints;

//         for( int i = 0; i < good_matches.size(); i++ ) {
//             //-- Get the keypoints from the good matches
//             img1Keypoints.push_back( FASTKeypoints1[ good_matches[i].queryIdx ].pt );
//             img2Keypoints.push_back( FASTKeypoints2[ good_matches[i].trainIdx ].pt );
//         }

//         cv::Mat H = cv::findHomography(img1Keypoints,img2Keypoints,CV_RANSAC);

//         //-- Get the corners from the image_1 ( the object to be "detected" )
//         std::vector<cv::Point2f> img1_corners(4);
//         img1_corners[0] = cv::Point(0,0); img1_corners[1] = cv::Point( firstImage.cols, 0 );
//         img1_corners[2] = cv::Point( firstImage.cols, firstImage.rows ); img1_corners[3] = cv::Point( 0, firstImage.rows );

//         std::vector<cv::Point2f> img2_corners(4);

//         if(drawMatches){
            
//         //std::cout<<"oi passou aqui"<<std::endl;
        
//         //cv::drawMatches( firstImage, FASTKeypoints1, secondImage, FASTKeypoints2,good_matches, img_matches,
//         //                 cv::Scalar::all(-1), cv::Scalar::all(-1),std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

//             if(!H.empty()){
//                 cv::perspectiveTransform( img1_corners, img2_corners, H);

            
//                 // superior esquerdo
//                 //filledCircle(secondImage, img2_corners[0], 10, cv::Scalar( 255, 0, 0));
                
//                 // superior direito
//                 //filledCircle(secondImage, img2_corners[1], 10, cv::Scalar( 255, 0, 0));
                
//                 // inferior direito
//                 //filledCircle(secondImage, img2_corners[2], 10, cv::Scalar( 255, 0, 0));

//                 // inferior esquerdo
//                 //filledCircle(secondImage, img2_corners[3], 10, cv::Scalar( 255, 0, 0));

                

//                 cv::line( secondImage, img2_corners[0] , img2_corners[1] , cv::Scalar( 0, 255, 0), 4 );
//                 cv::line( secondImage, img2_corners[1] , img2_corners[2] , cv::Scalar( 0, 255, 0), 4 );
//                 cv::line( secondImage, img2_corners[2] , img2_corners[3] , cv::Scalar( 0, 255, 0), 4 );
//                 cv::line( secondImage, img2_corners[3] , img2_corners[0] , cv::Scalar( 0, 255, 0), 4 );

//             }
//             else{
//                 //std::cout<<"No Homography"<<std::endl;
//             }

        
//         }

//         if(!H.empty()){
//             //std::cout<<"Homography Computed"<<std::endl;
//         }
//         return H;

//     }


    cv::Mat computeHomography(cv::Mat& Image, int frameIndex) {
        // Checking with Robust Matcher
        VO::RobustMatcher robustMatcher;

        int fast_threshold = 35; // quanto maior, menos features
        int points_retain = 175;


        bool nonmaxSuppression = true;

        if(frameIndex % 10 == 0) {
            firstImage = Image;
            
            cv::FAST(firstImage,FASTKeypoints1,fast_threshold,nonmaxSuppression);

            //std::cout<<"Num features 1: "<<FASTKeypoints1.size()<<std::endl;
            

            cv::KeyPointsFilter::retainBest(FASTKeypoints1, points_retain);

            KeyPoint::convert(FASTKeypoints1, points1, vector<int>());

            robustMatcher.computeDescriptors(firstImage,FASTKeypoints1,orbDescriptors1);

            if ( orbDescriptors1.empty() )
            cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);

            if(orbDescriptors1.type()!=CV_32F) {
                orbDescriptors1.convertTo(orbDescriptors1, CV_32F);
                orbDescriptors1.convertTo(orbDescriptors1_8U, CV_8U);
            }

        } else {
            secondImage = Image;
        }

        cv::FAST(secondImage,FASTKeypoints2,fast_threshold,nonmaxSuppression);
        //std::cout<<"Num features 2: "<<FASTKeypoints2.size()<<std::endl;

        //std::cout<<"Passou aqui 1"<<std::endl;
        
        // filtra os melhores 150 pontos
        // http://answers.opencv.org/question/12316/set-a-threshold-on-fast-feature-detection/
        cv::KeyPointsFilter::retainBest(FASTKeypoints2, points_retain);

        //std::cout<<"Passou aqui 2"<<std::endl;

        
          
        KeyPoint::convert(FASTKeypoints2, points2, vector<int>());

        

        
        robustMatcher.computeDescriptors(secondImage,FASTKeypoints2,orbDescriptors2);

        //std::cout<<"Passou aqui 3"<<std::endl;

        
        if ( orbDescriptors2.empty() )
            cvError(0,"MatchFinder"," 2nd descriptor empty",__FILE__,__LINE__);

        

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
        for( int i = 0; i < orbDescriptors1.rows; i++ ) { 
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

//            std::cout<<"Min Distance : "<<min_dist<<" Max Distance : "<<max_dist<<std::endl;

        std::vector< cv::DMatch > good_matches;

        for( int i = 0; i < orbDescriptors1.rows; i++ ) { 
            if( matches[i].distance <= cv::max(5*min_dist, 200.0) ) {
                good_matches.push_back( matches[i]);
            }
        }

        //-- Draw only "good" matches
        cv::Mat img_matches;


        //-- Localize the object
        std::vector<cv::Point2f> img1Keypoints;
        std::vector<cv::Point2f> img2Keypoints;

        for( int i = 0; i < good_matches.size(); i++ ) {
            //-- Get the keypoints from the good matches
            img1Keypoints.push_back( FASTKeypoints1[ good_matches[i].queryIdx ].pt );
            img2Keypoints.push_back( FASTKeypoints2[ good_matches[i].trainIdx ].pt );
        }

        cv::Mat H = cv::findHomography(img1Keypoints,img2Keypoints,CV_RANSAC);


        //cv::drawMatches( firstImage, FASTKeypoints1, secondImage, FASTKeypoints2,good_matches, img_matches,
        //                    cv::Scalar::all(-1), cv::Scalar::all(-1),std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        //-- Get the corners from the image_1 ( the object to be "detected" )
        std::vector<cv::Point2f> img1_corners(4);
        img1_corners[0] = cv::Point(0,0); 
        img1_corners[1] = cv::Point( firstImage.cols, 0 );
        img1_corners[2] = cv::Point( firstImage.cols, firstImage.rows ); 
        img1_corners[3] = cv::Point( 0, firstImage.rows );

        std::vector<cv::Point2f> img2_corners(4);

        if(!H.empty()){
            cv::perspectiveTransform( img1_corners, img2_corners, H);

            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            // cv::line( img_matches, img2_corners[0] + cv::Point2f( firstImage.cols, 0), img2_corners[1] + cv::Point2f( firstImage.cols, 0), cv::Scalar(0, 255, 0), 4 );
            // cv::line( img_matches, img2_corners[1] + cv::Point2f( firstImage.cols, 0), img2_corners[2] + cv::Point2f( firstImage.cols, 0), cv::Scalar( 0, 255, 0), 4 );
            // cv::line( img_matches, img2_corners[2] + cv::Point2f( firstImage.cols, 0), img2_corners[3] + cv::Point2f( firstImage.cols, 0), cv::Scalar( 0, 255, 0), 4 );
            // cv::line( img_matches, img2_corners[3] + cv::Point2f( firstImage.cols, 0), img2_corners[0] + cv::Point2f( firstImage.cols, 0), cv::Scalar( 0, 255, 0), 4 );
            cv::line( Image, img2_corners[0] , img2_corners[1] , cv::Scalar( 255, 255, 255), 4 );
            cv::line( Image, img2_corners[1] , img2_corners[2] , cv::Scalar( 255, 255, 255), 4 );
            cv::line( Image, img2_corners[2] , img2_corners[3] , cv::Scalar( 255, 255, 255), 4 );
            cv::line( Image, img2_corners[3] , img2_corners[0] , cv::Scalar( 255, 255, 255), 4 );
        
            //std::cout<<H<<std::endl;

            

            //std::cout<< "PROFUNDIDADE = "<< profundidade <<std::endl;
            //std::cout<< "LATERAL = "<< lateral <<std::endl;

        
        } else{
            std::cout<<"No Homography"<<std::endl;
        }


        if(!H.empty()){
            //std::cout<<"Homography Computed"<<std::endl;
        }

        return H;
    
    }

    


    // implementar o roi
    // http://funvision.blogspot.com.br/2015/12/basic-opencv-3-mat-tutorial-part-2-roi.html

   //////////////////////////////////////////////////////////////////////////
   double* EMSCRIPTEN_KEEPALIVE teste_match(  int width, 
                                            int height,
                                            cv::Vec4b* frame4b_ptr,
                                            cv::Vec4b* frame4b_ptr_out,
                                            int frameIndex,
                                            double *buf, int bufSize
                                        ) try { 

        
        
        
        
        Mat frame(height, width, CV_8UC4, frame4b_ptr);

        Mat gray(height, width, CV_8UC3, Scalar(0,0,0));

        Mat img_out(height, width, CV_8UC4, frame4b_ptr_out);

        // int roi_width = 320;
        // int roi_height = 240;

        // int roi_x = 0;
        // int roi_y = 0;





        // Rect RectangleToSelect(x,y, roi_width, roi_height);
        // Mat source;
        // Mat roiImage = source(RectangleToSelect);


        //mat_white.copyTo(frame_white);

        //cv::Mat frame_white(height, width, CV_8UC3, Scalar(255,255,255));



        cv::cvtColor(frame, gray, CV_RGBA2GRAY);


        cv::Mat homography;
        bool drawMatches = true;
        //homography = computeHomographyFromKeypoints(frame_base, gray, points_0, points_1, drawMatches);

        homography = computeHomography(gray, frameIndex);

        //http://docs.opencv.org/3.3.0/d6/d6d/tutorial_mat_the_basic_image_container.html
        //cout << "M = " << endl << " " << homography << endl << endl;



        // 0123
        // RGBA
        const Mat in_mats[] = {gray, img_out };
        constexpr int from_to[] = { 0,0, 1,1, 2,2 };
        mixChannels(in_mats, std::size(in_mats), &img_out, 1, from_to, std::size(from_to)/2);


        // PROFUNDIDADE
        double profundidade = homography.at<double>(1, 1);
        
        // LATERAL
        double lateral = homography.at<double>(0, 2);


        double values[bufSize];
        
        // for (int i=0; i<bufSize; i++) {
        //     values[i] = buf[i] * 1;
        // }


        values[0] = profundidade;
        values[1] = lateral;

        //std::cout<<"Passou aqui 1"<<std::endl;
    
        auto arrayPtr = &values[0];
        return arrayPtr;



    } catch (std::exception const& e) {
        printf("Exception thrown teste_match: %s\n", e.what());
        return 0;
    } catch (...) {
        printf("Unknown exception thrown teste_match!\n");
        return 0;
    }

    


    bool EMSCRIPTEN_KEEPALIVE teste_features(int width, 
                                            int height,
                                            cv::Vec4b* frame4b_ptr,
                                            cv::Vec4b* frame4b_ptr_out) try { 
    
        Mat rgba_in(height, width, CV_8UC4, frame4b_ptr);
        Mat rgba_out(height, width, CV_8UC4, frame4b_ptr_out);


        Mat gray(height, width, CV_8UC3, Scalar(0,0,0));

        cv::cvtColor(rgba_in, gray, CV_RGBA2GRAY);



        std::vector<Point2f> points1;

        vector<KeyPoint> keypoints_1;
        int fast_threshold = 30; // quanto maior o threshold, menor o número de pontos
        bool nonmaxSuppression = true;
        FAST(gray, keypoints_1, fast_threshold, nonmaxSuppression);
        KeyPoint::convert(keypoints_1, points1, vector<int>());


        printf("kwypoints size = %d \n", keypoints_1.size());


        drawKeypoints(gray, keypoints_1, gray, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
                                                                                //DRAW_RICH_KEYPOINTS
                                                                                //DRAW_OVER_OUTIMG  
        // 0123
        // RGBA

        const Mat in_mats[] = { gray, rgba_in };
        constexpr int from_to[] = { 0,0, 1,1, 2,2 };
        mixChannels(in_mats, std::size(in_mats), &rgba_out, 1, from_to, std::size(from_to)/2);
      
        return true;
                
    
    
    } catch (std::exception const& e) {
        printf("Exception thrown teste_features: %s\n", e.what());
        return false;
    } catch (...) {
        printf("Unknown exception thrown teste_features!\n");
        return false;
    }

    int EMSCRIPTEN_KEEPALIVE teste_return(int width) try { 

        return width + 2;



    } catch (std::exception const& e) {
        printf("Exception thrown teste_return: %s\n", e.what());
        return false;
    } catch (...) {
        printf("Unknown exception thrown teste_return!\n");
        return false;
    }

    
    float EMSCRIPTEN_KEEPALIVE addNums (float *buffer, int bufSize) {
        float total = 0;
    
        for (int i=0; i<bufSize; i++) {
            total+= buffer[i];
        }
    
        return total;
    }

    float* EMSCRIPTEN_KEEPALIVE doubleValues (float *buf, int bufSize) {
        
            float values[bufSize];
        
            for (int i=0; i<bufSize; i++) {
                values[i] = buf[i] * 2.4;
            }
        
            auto arrayPtr = &values[0];
            return arrayPtr;
        }


    bool EMSCRIPTEN_KEEPALIVE teste_soma(int a, int b) try { 


        printf("resultado = %d\n", a*b);

        return true;


    } catch (std::exception const& e) {
        printf("Exception thrown teste_soma: %s\n", e.what());
        return false;
    } catch (...) {
        printf("Unknown exception thrown teste_soma!\n");
        return false;
    }

    bool EMSCRIPTEN_KEEPALIVE teste_gray(int width, 
                                        int height,
                                        cv::Vec4b* frame4b_ptr,
                                        cv::Vec4b* frame4b_ptr_out,
                                        int indexFrame) try {

        Mat rgba_in(height, width, CV_8UC4, frame4b_ptr);
        Mat rgba_out(height, width, CV_8UC4, frame4b_ptr_out);


        Mat gray(height, width, CV_8UC3, Scalar(0,0,0));

        cv::cvtColor(rgba_in, gray, CV_RGBA2GRAY);

        // mix BGR + A (from input) => RGBA output


        // 0123
        // RGBA

        const Mat in_mats[] = { gray, rgba_in };
        constexpr int from_to[] = { 0,0, 1,1, 2,2 };
        mixChannels(in_mats, std::size(in_mats), &rgba_out, 1, from_to, std::size(from_to)/2);




        return true;




    } catch (std::exception const& e) {
        printf("Exception thrown teste_gray: %s\n", e.what());
        return false;
    } catch (...) {
        printf("Unknown exception thrown teste_gray!\n");
        return false;
    }


   





   bool EMSCRIPTEN_KEEPALIVE teste_blur(int width, int height,
                                           cv::Vec4b* frame4b_ptr,
                                           cv::Vec4b* frame4b_ptr_out, 
                                           int hsteps) try {


      // wrap memory pointers with proper cv::Mat images (no copies)
      cv::Mat4b rgba_in(height, width, frame4b_ptr);
      cv::Mat4b rgba_out(height, width, frame4b_ptr_out);

      


      blur(rgba_in, rgba_out, Size( 5, 5 ));

      return true;




   } catch (std::exception const& e) {
      printf("Exception thrown teste_blur: %s\n", e.what());
      return false;
   } catch (...) {
      printf("Unknown exception thrown teste_blur!\n");
      return false;
   }



   

   //////////////////////////////////////////////////////////////////////////

   void EMSCRIPTEN_KEEPALIVE release()
   {
        frame_base.release();
   }

   //////////////////////////////////////////////////////////////////////////
}