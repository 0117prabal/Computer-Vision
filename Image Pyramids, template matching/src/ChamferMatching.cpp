#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class ChamferMatching{

private:

	Mat image;
	Mat toFind;

	Mat grayImage;
	Mat grayToFind;

	int instancesToDetect;

public:

	ChamferMatching(string pathImage, string pathToFind, int instancesToDetect){

		image = imread(pathImage, 1);
		toFind = imread(pathToFind, 1);

		if(!image.data || !toFind.data){

			cout<<"Unable to read the images"<<endl;
		}

		grayImage = imread(pathImage, 0);
		grayToFind = imread(pathToFind, 0);

		this->instancesToDetect = instancesToDetect;
	}

	void detect(){

		Mat imageEdges;
		Mat imageDistanceTransform;

		GaussianBlur(grayImage, grayImage, cv::Size(3,3), 0);
		Canny(grayImage, imageEdges, 300, 400);
		threshold(imageEdges, imageEdges, 200, 255, CV_THRESH_BINARY_INV);
		distanceTransform(imageEdges, imageDistanceTransform, CV_DIST_L2, CV_DIST_MASK_PRECISE);
		
	    double maxTraffDT;
	    minMaxIdx(imageDistanceTransform, NULL, &maxTraffDT);
	    Mat imageDistanceTransformDisp;
    	imageDistanceTransform.convertTo(imageDistanceTransformDisp, CV_8UC1, 255/maxTraffDT );
		
		imshow("Distance transform", imageDistanceTransformDisp);
		waitKey(0);

		int imageRescaleFactor = 10;

		resize(grayToFind, grayToFind, Size(toFind.cols/imageRescaleFactor, toFind.rows / imageRescaleFactor));
		resize(toFind, toFind, Size(toFind.cols/imageRescaleFactor, toFind.rows / imageRescaleFactor));

		int check_Rows = toFind.rows - grayToFind.rows + 1;
	    int check_Cols = toFind.cols - grayToFind.cols + 1;
    	Mat detectionScores = Mat::zeros(check_Rows,check_Cols,CV_32F);

    	Mat detectionScoresDIST;

    	Mat resizedTemplateCONTOUR;
	    Mat resizedTemplateCONTOUR_DISP;
	    // Find Template Edges
	    Canny(grayToFind, resizedTemplateCONTOUR,200, 400);
	    // Conversion, in order to be used as kernel2D!
	    resizedTemplateCONTOUR.convertTo(resizedTemplateCONTOUR, CV_64F, 1);
	    // Prepare visualization & visualize
	    resizedTemplateCONTOUR.convertTo( resizedTemplateCONTOUR_DISP, CV_8UC1, 255);
	    imshow("image",resizedTemplateCONTOUR_DISP);
	    waitKey(0);

	    filter2D(imageDistanceTransform, detectionScores,-1,resizedTemplateCONTOUR, Point(0,0));

	    Point minPt,maxPt;
    	int off = 40;

	    Mat im_Traffic_Det = image.clone();
	    GaussianBlur( detectionScores, detectionScores, Size(0,0), 1.5);

	    double minVal, maxVal;

	    for (int inst=0; inst<instancesToDetect; inst++){

	     	minMaxLoc( detectionScores, &minVal, &maxVal, &minPt, &maxPt);

	     	if (inst==0){

	            detectionScores.convertTo( detectionScoresDIST, CV_8UC1, 255/maxVal );
	            // Depict vote map || note the black area after non-Maximum suppression (after the 1st detection)
	            std::cout << std::endl << "Image depicting Vote Map (interested in the lowest/darkest peak)" << std::endl << std::endl;
	            cv::imshow("Voting Space",detectionScoresDIST);
	            cv::waitKey(0);

        	}

        	toFind.copyTo( im_Traffic_Det(Rect(minPt.x, minPt.y, toFind.cols, toFind.rows)));
   		    Rect boundingBox(minPt.x, minPt.y, toFind.cols, toFind.rows ); // x y w h
        	rectangle(im_Traffic_Det, boundingBox, Scalar(0,255,0) ); // BGR

        	cv::Mat suppressor = cv::Mat::ones(2*off+1,2*off+1,CV_32FC1) * FLT_MAX;
	        suppressor.copyTo( detectionScores(cv::Rect(minPt.x-off, minPt.y-off, suppressor.cols, suppressor.rows)));

	     }

	      cv::imshow("Final detections", im_Traffic_Det);
	      std::cout << "Image with Template Detections-Overlay" << std::endl << std::endl;
    	  cv::waitKey(0);

	}

	string type2str(int type) {
	  
	  string r;

	  uchar depth = type & CV_MAT_DEPTH_MASK;
	  uchar chans = 1 + (type >> CV_CN_SHIFT);

	  switch ( depth ) {
	    case CV_8U:  r = "8U"; break;
	    case CV_8S:  r = "8S"; break;
	    case CV_16U: r = "16U"; break;
	    case CV_16S: r = "16S"; break;
	    case CV_32S: r = "32S"; break;
	    case CV_32F: r = "32F"; break;
	    case CV_64F: r = "64F"; break;
	    default:     r = "User"; break;
	  }

	  r += "C";
	  r += (chans+'0');

	  return r;

}

	~ChamferMatching(){

		//nothing to do here

	}


};