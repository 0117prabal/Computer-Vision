#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Parser.cpp"
#include "EpipolarGeometry.cpp"
#include "DisparityMap.cpp"
#include "ImageRectification.cpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
/*
	Parser parser("images/corresp.txt");
	parser.parse();

	EpipolarGeometry ep(parser.getImageOnePoints(), parser.getImageTwoPoints());
	ep.calculateFundamentalMatrix();
	ep.visualize();

	DisparityMap map;
	map.run();*/

	ImageRectification rectifier;
	rectifier.rectify();



    return 0;
}
