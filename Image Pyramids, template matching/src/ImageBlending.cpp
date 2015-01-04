#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Pyramids.cpp"

using namespace std;
using namespace cv;

class ImageBlending{

private:

	Mat image1, image2;
	Mat blendedImage;

public:

	ImageBlending(string path1, string path2){

		image1 = imread(path1, 1);

		image2 = imread(path2, 1);

		if(!image1.data || !image2.data){

			cout<<"Unable to read the image"<<endl;

		}

		image1.convertTo(image1, CV_32FC3, (1./255.));
		image2.convertTo(image2, CV_32FC3, (1./255.));

		imshow("Image1", image1);
		waitKey(0);

		imshow("Image2", image2);
		waitKey(0);
	}


	void showBlendedImage(){

		vector<Mat> maskPyramid;
		vector<Mat> applePyramid;
		vector<Mat> orangePyramid;

		Mat mask = Mat::zeros(image1.size(), CV_32F);
		rectangle(mask, Rect(0, 0, mask.cols/2, mask.rows), Scalar::all(1.), CV_FILLED);

		buildPyramid(mask, maskPyramid, 3);
		buildPyramid(image1, applePyramid, 3);
		buildPyramid(image2, orangePyramid, 3);

		blendedImage.create(maskPyramid[3].size(), image1.type());

		for (int i = 0 ; i < blendedImage.rows; i++){

	        for (int j = 0 ; j < blendedImage.cols; j++){


	            blendedImage.at<Vec<float,3> >(i,j) = maskPyramid[3].at<float>(i,j) * applePyramid[3].at<Vec<float,3> >(i,j) + (1.-maskPyramid[3].at<float>(i,j)) * orangePyramid[3].at<Vec<float,3> >(i,j);

    	    }

 	   }

 	   for(int i = 3 ; i > 0 ; i--){

 	   	pyrUp(blendedImage, blendedImage, maskPyramid[i-1].size());
 	   	Mat aLap, oLap;
 	   	pyrUp(applePyramid[i], aLap, applePyramid[i-1].size());
        subtract(applePyramid[i-1], aLap, aLap, noArray(), aLap.type());
        pyrUp(orangePyramid[i], oLap, orangePyramid[i-1].size());
        subtract(orangePyramid[i-1], oLap, oLap, noArray(), oLap.type());

        for (int x=0; x<blendedImage.cols; ++x)
        {
            for (int y=0; y<blendedImage.rows; ++y)
            {
                blendedImage.at<Vec<float,3> >(y,x) += maskPyramid[i-1].at<float>(y,x) * aLap.at<Vec<float,3> >(y,x)
                                                     + (1.-maskPyramid[i-1].at<float>(y,x)) * oLap.at<Vec<float,3> >(y,x);
            }
        }

 	   }

 	   imshow("Blended image", blendedImage);
       waitKey(0);

	}

	~ImageBlending(){

		//nothing to do here
	}
};