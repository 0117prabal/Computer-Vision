#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include "GMEM.cpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv){

	GMEM gm("images/gnome.png");
	gm.run();

	return 0;
}