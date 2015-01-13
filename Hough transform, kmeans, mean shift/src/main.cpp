#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "HoughTransform.cpp"
#include "KMeans.cpp"
#include "MeanShift.cpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv){

	/*HoughTransform ht("images/circles.png");
	ht.run();

	KMeans kmeans("images/flower.png");
	kmeans.run();*/

	MeanShift shift("images/flower.png");
	shift.run();

	return 0;
}