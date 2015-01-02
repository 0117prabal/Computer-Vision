#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "IntegralImages.cpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv){

	IntegralImages integral("images/bonn.png");
	integral.computeIntensities();

	return 0;
}
