#include <stdio.h>
#include <iostream>
#include <string>

#include <emscripten.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"


#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include "src/VO.h"



using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


// global data

cv::Mat firstImage(120, 160, CV_8UC3, Scalar(0,0,0));
cv::Mat secondImage(120, 160, CV_8UC3, Scalar(0,0,0));

// cv::Mat firstImage(240, 320, CV_8UC3, Scalar(0,0,0));
// cv::Mat secondImage(240, 320, CV_8UC3, Scalar(0,0,0));

std::vector<cv::KeyPoint> FASTKeypoints1,FASTKeypoints2;
cv::Mat orbDescriptors1,orbDescriptors2;
cv::Mat orbDescriptors1_8U,orbDescriptors2_8U;

vector<Point2f> points1;
vector<Point2f> points2;

extern "C" 
{
  
    cv::Mat computeHomography(cv::Mat& Image, int frameIndex) {

        VO::RobustMatcher robustMatcher;
        // limiar de intensidade do pixel , para ser considerado um canto (ponto característico)
        int fast_threshold = 20; 
        // selecionar os melhores x pontos
        int points_retain = 150; // 
        bool nonmaxSuppression = true;

        // seleciona um frame base a cada 15
        if(frameIndex % 15 == 0) {
            
            firstImage = Image;

            // extrai os pontos com o algorimo FAST
            cv::FAST(firstImage, FASTKeypoints1, fast_threshold, nonmaxSuppression);

            // seleciona os melhores pontos encontrados
            cv::KeyPointsFilter::retainBest(FASTKeypoints1, points_retain);

            // esta função utiliza o ORB para descrever os pontos
            robustMatcher.computeDescriptors(firstImage,FASTKeypoints1,orbDescriptors1);

            if ( orbDescriptors1.empty() )
            cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);

            if(orbDescriptors1.type()!=CV_32F) {
                orbDescriptors1.convertTo(orbDescriptors1, CV_32F);
                orbDescriptors1.convertTo(orbDescriptors1_8U, CV_8U);
            }

        } else {
            
            // se não é o frame base
            secondImage = Image;
        }

        cv::FAST(secondImage,FASTKeypoints2,fast_threshold,nonmaxSuppression);
        
        // filtra os melhores 150 pontos
        cv::KeyPointsFilter::retainBest(FASTKeypoints2, points_retain);

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



        double max_dist = 0; double min_dist = 2000; // era 2000

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

        std::vector<cv::Point2f> img1Keypoints;
        std::vector<cv::Point2f> img2Keypoints;

        for( int i = 0; i < good_matches.size(); i++ ) {
            // extrai os pontos a partir das melhores associações
            img1Keypoints.push_back( FASTKeypoints1[ good_matches[i].queryIdx ].pt );
            img2Keypoints.push_back( FASTKeypoints2[ good_matches[i].trainIdx ].pt );
        }

        cv::Mat H = cv::findHomography(img1Keypoints,img2Keypoints,CV_RANSAC);

        if(!H.empty()){
            //std::cout<<"Homography Computed"<<std::endl;
        } else {
            std::cout<<" NO Homography "<<std::endl;
        }

        return H;
    
    }

    

   //////////////////////////////////////////////////////////////////////////
   double* EMSCRIPTEN_KEEPALIVE vo_homography(  int width, 
                                                int height,
                                                cv::Vec4b* frame4b_ptr,
                                                int frameIndex,
                                                int matrix_size
                                        ) try { 
        // cria uma matriz representado o frame, basada no ponteiro enviado
        Mat frame(height, width, CV_8UC4, frame4b_ptr);
        // matriz que recebe a imagem convertida para escala de cinza
        Mat gray(height, width, CV_8UC3, Scalar(0,0,0));
        // converte a imagem em escala de cinza
        cv::cvtColor(frame, gray, CV_RGBA2GRAY); 
        // chama a função responsável por computr a homografia
        cv::Mat homography;
        homography = computeHomography(gray, frameIndex);

        // cria um vetor que reacebe os valores da matriz de homografia
        double values[matrix_size];
        values[0] = homography.at<double>(0, 0);
        values[1] = homography.at<double>(0, 1);
        values[2] = homography.at<double>(0, 2);
        values[3] = homography.at<double>(1, 0);
        values[4] = homography.at<double>(1, 1);
        values[5] = homography.at<double>(1, 2);
        values[6] = homography.at<double>(2, 0);
        values[7] = homography.at<double>(2, 1);
        values[8] = homography.at<double>(2, 2);

        // retorna o ponteiro para o vetor contendo os valores da homografia
        auto arrayPtr = &values[0];
        return arrayPtr;

    } catch (std::exception const& e) {
        printf("Exception thrown vo_homography: %s\n", e.what());
        return 0;
    } catch (...) {
        printf("Unknown exception thrown vo_homography!\n");
        return 0;
    }


    int EMSCRIPTEN_KEEPALIVE teste_soma(int a, int b) try { 


        return a + b;


    } catch (std::exception const& e) {
        printf("Exception thrown teste_soma: %s\n", e.what());
        return false;
    } catch (...) {
        printf("Unknown exception thrown teste_soma!\n");
        return false;
    }



}