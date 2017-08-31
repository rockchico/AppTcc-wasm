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



using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


// global data
cv::Mat3b bgr_g, bgr_out_g;



extern "C" 
{
    void featureDetection(Mat& img, vector<Point2f>& points)	{   //uses FAST as of now, modify parameters as necessary
        vector<KeyPoint> keypoints;
        int fast_threshold = 40;
        bool nonmaxSuppression = true;
        FAST(img, keypoints, fast_threshold, nonmaxSuppression);
        KeyPoint::convert(keypoints, points, vector<int>());

        drawKeypoints(img, keypoints, img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    }

    void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{ 
        
        //this function automatically gets rid of points for which tracking fails
    
        vector<float> err;					
        Size winSize=Size(21,21);																								
        TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    
        calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
    
        //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
        int indexCorrection = 0;
        for( int i=0; i<status.size(); i++)
            {  Point2f pt = points2.at(i- indexCorrection);
                if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
                    if((pt.x<0)||(pt.y<0))	{
                        status.at(i) = 0;
                    }
                    points1.erase (points1.begin() + (i - indexCorrection));
                    points2.erase (points2.begin() + (i - indexCorrection));
                    indexCorrection++;
                }
    
            }
    
    }
    



   //////////////////////////////////////////////////////////////////////////
    bool EMSCRIPTEN_KEEPALIVE teste_match(  int width, 
                                            int height,
                                            cv::Vec4b* frame_0_ptr,
                                            cv::Vec4b* frame_1_ptr,
                                            cv::Vec4b* frame4b_ptr_out) try { 

        Mat frame_0(height, width, CV_8UC4, frame_0_ptr);
        Mat frame_1(height, width, CV_8UC4, frame_1_ptr);

        Mat gray_0(height, width, CV_8UC3, Scalar(0,0,0));
        Mat gray_1(height, width, CV_8UC3, Scalar(0,0,0));

        Mat rgba_out(height, width, CV_8UC4, frame4b_ptr_out);

        cv::cvtColor(frame_0, gray_0, CV_RGBA2GRAY);
        cv::cvtColor(frame_1, gray_1, CV_RGBA2GRAY);

        // feature detection, tracking
        std::vector<Point2f> points_0, points_1;        //vectors to store the coordinates of the feature points
        featureDetection(gray_0, points_0);        //detect features in img_1

        vector<uchar> status;
        //featureTracking(gray_0, gray_1, points_0, points_1, status); //track those features to img_2



        //Mat rgba_out(height, width, CV_8UC4, frame4b_ptr_out);


        
        // 0123
        // RGBA

        const Mat in_mats[] = { gray_0, frame_0 };
        constexpr int from_to[] = { 0,0, 1,1, 2,2 };
        mixChannels(in_mats, std::size(in_mats), &rgba_out, 1, from_to, std::size(from_to)/2);

        return true;



    } catch (std::exception const& e) {
        printf("Exception thrown teste_match: %s\n", e.what());
        return false;
    } catch (...) {
        printf("Unknown exception thrown teste_match!\n");
        return false;
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
      bgr_g.release();
      bgr_out_g.release();
   }

   //////////////////////////////////////////////////////////////////////////
}