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


#include "VisualOdometry/include/VO.h"



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


   //////////////////////////////////////////////////////////////////////////
   void EMSCRIPTEN_KEEPALIVE teste_vo(  int width, 
                                            int height,
                                            cv::Vec4b* frame4b_ptr,
                                            cv::Vec4b* frame4b_ptr_out,
                                            int frameIndex,
                                            double *buf, int bufSize
                                        ) try { 

        
        Mat frame(height, width, CV_8UC4, frame4b_ptr);
        Mat gray(height, width, CV_8UC3, Scalar(0,0,0));
        Mat img_out(height, width, CV_8UC4, frame4b_ptr_out);

        //cv::cvtColor(frame, gray, CV_RGBA2GRAY);

        bool enableHomography = true;
        bool drawMatches = false;  // relevant only if enableHomography is true


        float cameraIntrinsicMatrix[3][3];

        // 1.2	0	3.5
        // 0	2.5	3.6
        // 0	0	1   
        
        cameraIntrinsicMatrix[0][0] = 1.2;
        cameraIntrinsicMatrix[0][1] = 0;
        cameraIntrinsicMatrix[0][2] = 3.5;
        cameraIntrinsicMatrix[1][0] = 0;
        cameraIntrinsicMatrix[1][1] = 2.5;
        cameraIntrinsicMatrix[1][2] = 3.6;
        cameraIntrinsicMatrix[2][0] = 0;
        cameraIntrinsicMatrix[2][1] = 0;
        cameraIntrinsicMatrix[2][2] = 1;
        


        cv::Mat cameraMatrix;

        cameraMatrix = cv::Mat(3,3,CV_32FC1,&cameraIntrinsicMatrix);

        VO::featureOperations voModule(frame, cameraMatrix,drawMatches,enableHomography, frameIndex);


        
        

  
        const Mat in_mats[] = {frame, img_out };
        constexpr int from_to[] = { 0,0, 1,1, 2,2 };
        mixChannels(in_mats, std::size(in_mats), &img_out, 1, from_to, std::size(from_to)/2);


    } catch (std::exception const& e) {
        printf("Exception thrown teste_match: %s\n", e.what());
        //return 0;
    } catch (...) {
        printf("Unknown exception thrown teste_match!\n");
        //  return 0;
    }

    

   

   //////////////////////////////////////////////////////////////////////////

   void EMSCRIPTEN_KEEPALIVE release()
   {
        frame_base.release();
   }

   //////////////////////////////////////////////////////////////////////////
}