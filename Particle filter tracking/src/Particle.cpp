#ifndef PARTICLE_CPP
#define PARTICLE_CPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

class Particle{

public:

	cv::MatND hist; // histogram of object, as given by (1) the initializinfg Bounding Box OR (2) the detection @previousframe
    cv::Mat   roi;
    int       xxx;  // coordinate
    int       yyy;  // coordinate
    int       www;  // width
    int       hhh;  // height
    double    weight;

    double  weightNormalized;
    double  arcCircleStart360;

    void show_onFrame(cv::Mat &currFrameSHOW, bool shouldImShow) {

        cv::Rect rectt(xxx,yyy,www,hhh);
        cv::rectangle(currFrameSHOW,rectt,cv::Scalar(0,255,0),2);
        if (shouldImShow){
        	cv::imshow("Detection",currFrameSHOW);
        }

    }

    void initializeManually(cv::Mat &currFrame, int xxxIN, int yyyIN, int wwwIN, int hhhIN){

	    xxx    = xxxIN;
	    yyy    = yyyIN;
	    www    = wwwIN;
	    hhh    = hhhIN;
	    weight = -888; // reset to 'default' dummy value

	    cv::Rect  rectt(xxx,yyy,www,hhh);
	    currFrame(rectt).copyTo( roi );

	    calculateHistogramm();
	}

	void calculateHistogramm(){

	    int          channels[] = {0,1,2};
	    float        rangeRRR[] = {0,256}; // {0,256} = 0..255
	    float        rangeGGG[] = {0,256}; // {0,256} = 0..255
	    float        rangeBBB[] = {0,256}; // {0,256} = 0..255
	    const float* ranges[  ] = {rangeBBB,rangeGGG,rangeRRR};
	    int          histSize[] = {16,16,16};

	    cv::calcHist(&roi,1,channels,cv::Mat(),hist,3,histSize,ranges,true,false);
	}

};

#endif