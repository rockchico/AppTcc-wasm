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




#include "color_cycle.h"




using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


// global data
cv::Mat3b bgr_g, bgr_out_g;



extern "C" 
{

    void MyLine( Mat img, Point start, Point end )
    {
      int thickness = 2;
      int lineType = 8;
      line( img,
            start,
            end,
            Scalar( 0, 0, 0 ),
            thickness,
            lineType );
    }
    
    void MyEllipse( Mat img, double angle )
    {
      int thickness = 2;
      int lineType = 8;
    
      int w = 100;
    
      ellipse( img,
               Point( w/2.0, w/2.0 ),
               Size( w/4.0, w/16.0 ),
               angle,
               0,
               360,
               Scalar( 255, 0, 0 ),
               thickness,
               lineType );
    }
    
    void MyFilledCircle( Mat img, Point center, int radius, cv::Scalar cor )
    {
     int thickness = -1;
     int lineType = 8;
    
     circle( img,
             center,
             radius,
             cor,
             thickness,
             lineType );
    }



   //////////////////////////////////////////////////////////////////////////
   

    int EMSCRIPTEN_KEEPALIVE teste_return(int width) try { 

       

        return width + 2;



    } catch (std::exception const& e) {
        printf("Exception thrown teste_return: %s\n", e.what());
        return false;
    } catch (...) {
        printf("Unknown exception thrown teste_return!\n");
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



   bool EMSCRIPTEN_KEEPALIVE rotate_colors(int width, int height,
                                           cv::Vec4b* frame4b_ptr,
                                           cv::Vec4b* frame4b_ptr_out, 
                                           int hsteps) try
   {
      // wrap memory pointers with proper cv::Mat images (no copies)
      cv::Mat4b rgba_in(height, width, frame4b_ptr);
      cv::Mat4b rgba_out(height, width, frame4b_ptr_out);

      // allocate 3-channel images if needed
      bgr_g.create(rgba_in.size());
      bgr_out_g.create(rgba_in.size());

      cv::cvtColor(rgba_in, bgr_g, CV_RGBA2BGR);
      color_cycle::rotate_hue(bgr_g, bgr_out_g, hsteps);

      // mix BGR + A (from input) => RGBA output
      const Mat in_mats[] = { bgr_out_g, rgba_in };
      constexpr int from_to[] = { 0,2, 1,1, 2,0, 6,3 };
      mixChannels(in_mats, std::size(in_mats), &rgba_out, 1, from_to, std::size(from_to)/2);
      return true;

   }
   catch (std::exception const& e)
   {
      printf("Exception thrown: %s\n", e.what());
      return false;
   }
   catch (...)
   {
      printf("Unknown exception thrown!\n");
      return false;
   }

   //////////////////////////////////////////////////////////////////////////

   void EMSCRIPTEN_KEEPALIVE release()
   {
      color_cycle::clear_all();
      bgr_g.release();
      bgr_out_g.release();
   }

   //////////////////////////////////////////////////////////////////////////
}