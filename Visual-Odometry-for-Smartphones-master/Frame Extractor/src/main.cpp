#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <fstream>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/date_formatting.hpp>
#include <boost/date_time/gregorian/greg_month.hpp>
#include "boost/date_time/posix_time/posix_time.hpp"
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <locale>
#include <string>
#include <sstream>


std::wstring FormatTime(boost::posix_time::ptime now)
{
  using namespace boost::posix_time;
  static std::locale loc(std::wcout.getloc(),
                         new wtime_facet(L"%Y%m%d_%H%M%S"));

  std::basic_stringstream<wchar_t> wss;
  wss.imbue(loc);
  wss << now;
  return wss.str();
}

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

int main( int argc, char** argv )
{
    cv::namedWindow("DisplayVideo", CV_WINDOW_AUTOSIZE );
    cv::VideoCapture cap(argv[1]);
    if ( !cap.isOpened() )  // if not success, exit program
    {
         std::cout << "Cannot open the video file." << std::endl;
         return -1;
    }

    boost::posix_time::ptime now;
    for(int i=0;i<1000000;i++){

    }

    cv::Mat frame;
    int num = 0;
    char s[20];
    while(1) {
        num++;
        bool bSuccess = cap.read(frame);
        now = boost::posix_time::microsec_clock::universal_time();
        std::wstring timeStamp(FormatTime(now));
        std::wcout << timeStamp << std::endl;
        const std::string currentTime( timeStamp.begin(),  timeStamp.end());
        std::string folderPath = "/home/vikiboy/Videos/Dataset/newDataset/";
        std::stringstream filePath ;
        filePath << folderPath << "frame_" << currentTime << "-" <<num<<".png";

        if (!bSuccess) //if not success, break loop
        {
           std::cout << "Cannot read the frame from video file" << std::endl;
           break;
        }
        cv::imshow( "DisplayVideo", frame );
        char c = cv::waitKey(33);
        if( c == 27 ) break;
        cv::imwrite(filePath.str(),frame);

    }
    cv::destroyAllWindows();
    return 0;
}
