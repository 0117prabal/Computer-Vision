#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "GMEM.cpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv){

	GMEM gmem("images/gnome.png");
	gmem.run();
	
	return 0;
}