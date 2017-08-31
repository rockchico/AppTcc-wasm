#pragma once

#include <opencv2/core.hpp>

namespace color_cycle
{
   void rotate_hue(cv::Mat3b const& img, cv::Mat3b& result_img, int hsteps = 10);
   void clear_all();
}
