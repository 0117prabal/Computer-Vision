#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "IntegralImages.cpp"
#include "HistogramEqualization.cpp"
#include "Filtering.cpp"
#include "SeparableFilters.cpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv){

	IntegralImages integral("images/bonn.png");
	integral.computeIntensities();

	HistogramEqualization hist("images/bonn.png");
	hist.run();

	Filtering filter("images/bonn.png");
	filter.run();

	SeparableFilters s("images/bonn.png");
	s.run();

	return 0;
}
