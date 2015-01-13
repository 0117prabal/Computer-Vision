#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class MeanShift{

private:

	Mat image;

public:

	MeanShift(string path){

		image = imread(path, 1);

		if(!image.data){

			cout<<"Unable to read the image"<<endl;

		}

		imshow("Original", image);
		waitKey(0);

	}

	void run(){

		Mat image_luv;
    	cvtColor(image, image_luv, CV_BGR2Luv);

    	Mat_<Vec2i> shiftCoord(image_luv.size());
	    const float varColor = 49.f;
	    const float varSpace = 81.f;
	    const int kwidth = ceil(sqrt(varSpace));

		    // for each pixel, compute end point (pixel) of shift vector
	    for (int y=0; y<shiftCoord.rows; ++y) {
	        for (int x=0; x<shiftCoord.cols; ++x) {

	            cv::Vec2i& sC = shiftCoord(y,x);
	          //cv::Vec2i& sC = shiftCoord.at<cv::Vec2i>(y,x);

	            float xShift=0.f, yShift=0.f;
	            float sum=0.f, w;
	            int r,c;

	            const Vec3f& centerColor = image_luv.at<cv::Vec3f>(y, x);

	        }
	    }

	}

	~MeanShift(){

	}

};