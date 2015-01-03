#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Pyramids.cpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv){

	Pyramids pyramids("images/traffic.jpg", 3);

	return 0;
}