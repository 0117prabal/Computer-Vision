#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Snakes.cpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv){


	Snakes snakes("images/ball.png");
	snakes.run();

	return 0;
}
