#include "VO.h"

bool enableHomography = true;
bool drawMatches = false;  // relevant only if enableHomography is true

int main(int argc,char* argv[])
{
    // Loading the Camera Intrinsic Matrix

    // Checking mode of operation

    if(argc==2){
        std::cout<<"******Visual Odometry using Camera******"<<std::endl;
        VO::featureOperations voModule(readCameraIntrinsic(argv[1]),drawMatches,enableHomography);
    }
    else if(argc==3){
        std::cout<<"******Visual Odometry from images in a directory****"<<std::endl;

        //Checking if the folder exists.

        if(!boost::filesystem::exists(argv[1])){
            std::cout<<"The specified folderpath doesn't exist."<<std::endl;
            std::cout<<"Terminating...."<<std::endl;
            return -1;
        }
        std::cout<<"\n \n"<<std::endl;
        VO::featureOperations voModule(argv[1],readCameraIntrinsic(argv[2]),drawMatches,enableHomography);
    }
    else if(argc!=2 && argc!=3){
        std::cout<<"Correct Usage : './VisualOdometry [Path to the folder containing images] [Path to Camera Matrix]' or './VisualOdometry [Path to Camera Matrix]' "<<std::endl;
        std::cout<<"Use the latter for loading images from a camera"<<std::endl;
        return -1;
    }
    return 0;
}


