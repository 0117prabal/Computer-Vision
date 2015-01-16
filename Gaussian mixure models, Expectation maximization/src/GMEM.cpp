#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class GMEM{

private:

	Mat image;

public:

	GMEM(string path){

		image = imread(path, 1);

		if(!image.data){

			cout<<"Unable to read the image"<<endl;

		}

	}

	void run(){

		const cv::Rect rrr(92,65,105,296);
	    cv::Mat mask = cv::Mat(image.size(), CV_64FC1, cv::Scalar::all(0));
	    cv::rectangle(mask, rrr, cv::Scalar::all(1), CV_FILLED);
	    /////////////////////////////////////////////////////////////////////////
	    cv::Mat           viz;
	    image.copyTo( viz);
	    rectangle(viz, rrr, cv::Scalar(0,0,255));
	    cv::imshow("gnome with rectangle", viz);
	    waitKey(0);

	    GMMs_withOpenCV(image, mask);
	}


	void GMMs_withOpenCV( const cv::Mat& img, const cv::Mat& mask ){

	    cv::Mat all_samples, pos_samples, neg_samples;
	    std::cout << "Organize Samples" << std::endl;   
	    organizeSamples( img, mask, all_samples, pos_samples, neg_samples, true); // results: Mx3,Nx3 // each row is a sample

	    // train the models
	    const cv::TermCriteria quick(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 10, FLT_EPSILON);
	    cv::EM pos_model(cv::EM::DEFAULT_NCLUSTERS, cv::EM::COV_MAT_GENERIC, quick);
	    cv::EM neg_model(cv::EM::DEFAULT_NCLUSTERS, cv::EM::COV_MAT_GENERIC, quick);

	    std::cout << "Train Positive Samples" << std::endl;   pos_model.train( pos_samples );
	    std::cout << "Train Negative Samples" << std::endl;   neg_model.train( neg_samples );

		//GMMs_withOpenCV_printEMinfo( pos_model, neg_model );

	    // classify the image pixels
	    std::cout << "Calculate Probabilities" << std::endl;
	    ////////////////////////////////////////
	    cv::Mat            neg_prob,  pos_prob;
	    predictImage( img, pos_model, pos_prob);        // pos likelihood logarithm value
	    predictImage( img, neg_model, neg_prob);        // neg likelihood logarithm value
	    ////////////////////////////////////////
	    cv::exp( pos_prob, pos_prob );                  // log likelihood -> likelihood
	    cv::exp( neg_prob, neg_prob );                  // log likelihood -> likelihood
	    //////////////////////////////
	    pos_prob = pos_prob / (pos_prob + neg_prob);    // likelihood -> probability (pos)
	  //neg_prob = 1 - pos_prob;                        //               probability (neg)

	    showGray( pos_prob,"positive probability - OpenCV",1 );
	  //showGray( neg_prob,"negative probability - OpenCV",0 );

	    cv::waitKey();
	}

	~GMEM(){

		//nothing to do here
	}

};