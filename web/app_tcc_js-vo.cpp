/*
    homografia
    http://www.docs.opencv.org/2.4.13/doc/tutorials/features2d/feature_homography/feature_homography.html

    mono-vo
    https://github.com/avisingh599/mono-vo
*/




#include <stdio.h>
#include <iostream>
#include <string>
#include <queue>          // std::queue

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


//#include "VO.h"



using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


// global data

// cv::Mat frame_base(240, 320, CV_8UC3, Scalar(0,0,0));

// cv::Mat mat_white(240, 320, CV_8UC3, Scalar(255,255,255));
//cv::Mat frame_white(240, 320, CV_8UC3, Scalar(255,255,255));

cv::Mat firstImage_g(240, 320, CV_8UC3, Scalar(0,0,0));
cv::Mat secondImage_g(240, 320, CV_8UC3, Scalar(0,0,0));

cv::Mat img_aux(240, 320, CV_8UC3, Scalar(0,0,0));

// cv::Mat firstImage_u(240, 320, CV_8UC3, Scalar(0,0,0));
// cv::Mat secondImage_u(240, 320, CV_8UC3, Scalar(0,0,0));

// std::vector<cv::KeyPoint> FASTKeypoints1,FASTKeypoints2;
// cv::Mat orbDescriptors1,orbDescriptors2;
// cv::Mat orbDescriptors1_8U,orbDescriptors2_8U;


std::queue<cv::Mat> frame_queue;

std::queue<double> teste_queue;

//Mat E, R, t, mask;
Mat R_f, t_f;

// incio no mapa
const int x_init = 300.0;
const int y_init = 300.0;



extern "C" 
{

std::vector<cv::Point2f> detectFeatures(cv::Mat img){
    cv::Mat distortCoeffs;
//    cv::Mat undistortedImg;

    //cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
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
    
bool trackFeatures(cv::Mat prevImg, cv::Mat currentImg, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, std::vector<uchar>& status){

    cv::Mat prevImg_gray,currentImg_gray;
    //cv::cvtColor(prevImg,prevImg_gray,CV_BGR2GRAY);
    //cv::cvtColor(currentImg,currentImg_gray,CV_BGR2GRAY);

    std::vector<float> err;
    cv::Size winSize=cv::Size(21,21);
    cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);


    cv::calcOpticalFlowPyrLK(prevImg, currentImg, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

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



bool cloned = false;
void featureOperations(cv::Mat firstImage, cv::Mat secondImage, cv::Mat cameraMatrix, bool drawMatches, bool enableHomography, int frameIndex){ // Loading images from a directory
        
    if ( !firstImage.data || !secondImage.data ) {
        std::cout<< " --(!) Error reading images " << std::endl;
    }


    std::vector<uchar> status;
    std::vector<cv::Point2f> points1,points2;

    points1 = detectFeatures(firstImage);

    // Change these accordingly. Intrinsics

    double focal = cameraMatrix.at<double>(0,0); // focal lengt visario
    Point2f pp(cameraMatrix.at<double>(0,2), cameraMatrix.at<double>(1,2)); //From intrinsical parameters matrix K

    // recovering the pose and the essential matrix
    cv::Mat E, R, t, mask;

    if(trackFeatures(firstImage,secondImage,points1,points2,status)) {
        E = cv::findEssentialMat(points2, points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
        cv::recoverPose(E, points2, points1, R, t, focal, pp, mask);
    }

    //cout << "R (default) = " << endl <<        R           << endl << endl;
    //cout << "t (default) = " << endl <<        t           << endl << endl;


    // cv::Mat prevImage = secondImage;
    // cv::Mat currImage;
    // std::vector<cv::Point2f> prevFeatures = points2;
    // std::vector<cv::Point2f> currFeatures;

    // if(cloned == false) {
    //     R_f = R.clone();
    //     t_f = t.clone();

    //     cloned = true;
    // } else {

    //     // scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));
    //     double scale = 1;

    //     // cout << "Scale is " << scale << endl;

    //     if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {

    //             t_f = t_f + scale * (R_f * t);
    //             R_f = R * R_f;

    //     }



    // }


    

//     cv::namedWindow("Camera",CV_WINDOW_AUTOSIZE);
//     cv::Mat trajectory = cv::Mat::zeros(600,600,CV_8UC3);

//     // The image stream from the 2nd image onwards
//     int fileIndex = 0;
//     std::string remainingImgs;
//     do {
//         ++fileIndex;
//         for(int i=0;i<10;i++){//Skipping 10 frames
//             *it++;
//         }
//         std::cout<<"\nFrame # : "<<fileIndex<<std::endl;
//         remainingImgs= ((*it).second).string();
//         currImage = cv::imread(remainingImgs,CV_LOAD_IMAGE_ANYCOLOR);
//         cv::resize(currImage,currImage,cv::Size(640,320));


//         // For FAST
//         if(this->trackFeatures(prevImage,currImage,prevFeatures,currFeatures,status)){
//             E = cv::findEssentialMat(currFeatures, prevFeatures, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
//             cv::recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);
//         }

//         // For Homography
//         if(enableHomography){
//             if(!(prevFeatures.empty()||currFeatures.empty())){
//             homography=this->computeHomographyFromKeypoints(prevImage,currImage,prevFeatures,currFeatures);
// //                homography=this->computeHomography(prevImage,currImage);
//             }
//         }

// //            scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));
//         double scale = 1;

//         //cout << "Scale is " << scale << endl;

//         if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {

//                 t_f = t_f + scale*(R_f*t);
//                 R_f = R*R_f;

//         }

//         // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
//         if (prevFeatures.size() < 1800)	{
//             prevFeatures=this->detectFeatures(prevImage);
//             this->trackFeatures(prevImage,currImage,prevFeatures,currFeatures, status);

//         }

//         prevImage = currImage.clone();
//         prevFeatures = currFeatures;

//         cv::imshow("Camera", currImage);
//         cv::waitKey(30);
//         if (prevFeatures.size() > 200){
//             if(this->plotTrajectory(trajectory,t_f,"Trajectory")){
//                 std::cout<<"Trajectory Plotted"<<std::endl;
//             }
//         }
//     } while ( comparator((*it++).first, last) );

//     cv::imwrite("Trajectory.png",trajectory);
        

    
}




   //////////////////////////////////////////////////////////////////////////
   double* EMSCRIPTEN_KEEPALIVE teste_vo(  int width, 
                                            int height,
                                            cv::Vec4b* frame4b_ptr,
                                            cv::Vec4b* frame4b_ptr_out,
                                            int frameIndex,
                                            double *buf, int bufSize
                                        ) try { 

        double x = 0.0;
        double y = 0.0;
        
        Mat frame(height, width, CV_8UC4, frame4b_ptr);
        Mat img_out(height, width, CV_8UC4, frame4b_ptr_out);

        double cameraIntrinsicMatrix[3][3];
        
        // 1.2	0	3.5
        // 0	2.5	3.6
        // 0	0	1   

        // cameraIntrinsicMatrix[0][0] = 1.2;
        // cameraIntrinsicMatrix[0][1] = 0;
        // cameraIntrinsicMatrix[0][2] = 3.5;
        // cameraIntrinsicMatrix[1][0] = 0;
        // cameraIntrinsicMatrix[1][1] = 2.5;
        // cameraIntrinsicMatrix[1][2] = 3.6;
        // cameraIntrinsicMatrix[2][0] = 0;
        // cameraIntrinsicMatrix[2][1] = 0;
        // cameraIntrinsicMatrix[2][2] = 1;


        // <kmatrix k00="226.4449768066" k10="0.0000000000" k20="165.5465698242" k01="0.0000000000" k11="211.3888702393" k21="120.4598159790" k02="0.0000000000" k12="0.0000000000" k22="1.0000000000"/>
 		// <distortion d0="0.0717896819" d1="-0.3373529315" d2="0.0000000000" d3="0.0000000000" d4="0.3692644835"/>

        cameraIntrinsicMatrix[0][0] = 226.4449768066;
        cameraIntrinsicMatrix[0][1] = 0;
        cameraIntrinsicMatrix[0][2] = 165.5465698242;
        cameraIntrinsicMatrix[1][0] = 0;
        cameraIntrinsicMatrix[1][1] = 211.3888702393;
        cameraIntrinsicMatrix[1][2] = 120.4598159790;
        cameraIntrinsicMatrix[2][0] = 0;
        cameraIntrinsicMatrix[2][1] = 0;
        cameraIntrinsicMatrix[2][2] = 1;

        cv::Mat cameraMatrix;
        cameraMatrix = cv::Mat(3, 3, CV_64F, &cameraIntrinsicMatrix);


        double distortCoeffs[5];
        distortCoeffs[0] = 0.0717896819;
        distortCoeffs[1] = -0.3373529315;
        distortCoeffs[2] = 0.0;
        distortCoeffs[3] = 0.0;
        distortCoeffs[4] = 0.3692644835;

        cv::Mat distortCoeffsMatrix;
                                        // row, col
        distortCoeffsMatrix = cv::Mat(5, 1, CV_64F, &distortCoeffs);

        // if(teste_queue.size() < 2) {
        //     teste_queue.push(frameIndex);
        // } else {
        //     x = teste_queue.front();

        //     teste_queue.pop();

        //     teste_queue.push(frameIndex);

        //     y = teste_queue.front();

        //     //teste_queue.pop();
        // }
 

        // std::cout<<"frame_queue size = "<< teste_queue.size() <<std::endl;


        if(frame_queue.size() < 2) {
            cv::cvtColor(frame, img_aux, CV_RGBA2GRAY);
            frame_queue.push(img_aux);   
        } else {

            // obtem o primeiro da fila
            firstImage_g = frame_queue.front();
            
            // tira o primeiro da fila
            frame_queue.pop();

            // insere u novo elemento no fim da fila
            cv::cvtColor(frame, img_aux, CV_RGBA2GRAY);
            frame_queue.push(img_aux);

            // obt;em o primeiro da fila, 
            secondImage_g = frame_queue.front();

            bool drawMatches = true;
            bool enableHomography = true;
            featureOperations(firstImage_g, secondImage_g, cameraMatrix, drawMatches, enableHomography, frameIndex);
            

        }


        //cout << "R_f (default) = " << endl <<        R_f           << endl << endl;
        //cout << "t_f (default) = " << endl <<        t_f           << endl << endl;

        

        //std::cout<<"frame_queue size"<< frame_queue.size() <<std::endl;


  
        const Mat in_mats[] = {firstImage_g, img_out };
        constexpr int from_to[] = { 0,0, 1,1, 2,2 };
        mixChannels(in_mats, std::size(in_mats), &img_out, 1, from_to, std::size(from_to)/2);


        double values[bufSize];
        
        // for (int i=0; i<bufSize; i++) {
        //     values[i] = buf[i] * 1;
        // }


        values[0] = x;
        values[1] = y;

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

    

   

   //////////////////////////////////////////////////////////////////////////

   void EMSCRIPTEN_KEEPALIVE release()
   {
        //frame_queue.release();
   }

   //////////////////////////////////////////////////////////////////////////
}