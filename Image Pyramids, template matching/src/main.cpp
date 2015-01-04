#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Pyramids.cpp"
#include "ImageBlending.cpp"
#include "ChamferMatching.cpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv){

	Pyramids pyramids("images/traffic.jpg", 3);

	ImageBlending blending("images/apple.jpg", "images/orange.jpg");
	blending.showBlendedImage();

	ChamferMatching matching("images/traffic.jpg", "images/sign.png", 2);
	matching.detect();

	return 0;
}