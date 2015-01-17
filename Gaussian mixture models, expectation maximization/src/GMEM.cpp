#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

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

		// create a mask for foreground/background separation
	    /////////////////////////////////////////////////////////////////////////
	    /////////////////////////////////////////////////////////////////////////
	    const cv::Rect rrr(92,65,105,296);
	    cv::Mat mask = cv::Mat(image.size(), CV_64FC1, cv::Scalar::all(0));
	    cv::rectangle(mask, rrr, cv::Scalar::all(1), CV_FILLED);
	    /////////////////////////////////////////////////////////////////////////
	    cv::Mat viz;
	    image.copyTo( viz);
	    rectangle(viz, rrr, cv::Scalar(0,0,255));
	    cv::imshow("gnome with rectangle", viz);
	    waitKey(0);

	    GMMs_withOpenCV(image, mask);

	    GMMs_Custom(image, mask );
	}

	void GMMs_Custom( const cv::Mat& gnome_INI, const cv::Mat& mask ){

	    cv::Mat                gnome_norm;
	    gnome_INI.convertTo(   gnome_norm, CV_32FC3, (1.0/255.0) ); // image normalized [0,255] -> [0,1]
	    gnome_norm.reshape( 1, gnome_norm.total() );

	    cv::Mat                                                                            all_samples, pos_samples, neg_samples;
	    std::cout << "Organize Samples" << std::endl;   organizeSamples( gnome_norm, mask, all_samples, pos_samples, neg_samples, false ); // results: Mx3,Nx3 // each row is a sample


	    cv::Mat                  GMM_Mixtures_Means_pos;           // filled in GMM_Initialize_with_KMeans  // CV_64FC1             Mx3 - M number of mixtures
	    cv::Mat                  GMM_Mixtures_Means_neg;           // filled in GMM_Initialize_with_KMeans  // CV_64FC1             Mx3
	    std::vector< cv::Mat >   GMM_Mixtures_Covs_pos;            // filled in GMM_Initialize_with_KMeans  // CV_64FC1  each mat ~ 3x3
	    std::vector< cv::Mat >   GMM_Mixtures_Covs_neg;            // filled in GMM_Initialize_with_KMeans  // CV_64FC1  each mat ~ 3x3
	    std::vector< cv::Mat >   GMM_Mixtures_Covs_pos_INV;        // filled in invert_3x3s                 // CV_64FC1  each mat ~ 3x3
	    std::vector< cv::Mat >   GMM_Mixtures_Covs_neg_INV;        // filled in invert_3x3s                 // CV_64FC1  each mat ~ 3x3
	    std::vector< double  >   GMM_Mixtures_Covs_pos_SQRT_DET;   // filled in sqrt_det                    // CV_64FC1  each mat ~ 3x3
	    std::vector< double  >   GMM_Mixtures_Covs_neg_SQRT_DET;   // filled in sqrt_det                    // CV_64FC1  each mat ~ 3x3
	    cv::Mat                  GMM_Mixtures_Weights_pos;         // filled in GMM_Train                   // CV_64FC1
	    cv::Mat                  GMM_Mixtures_Weights_neg;         // filled in GMM_Train                   // CV_64FC1

	    /////////////////////////////////////////
	    // initialize with kmeans (with color) //
	    /////////////////////////////////////////

	    ///////////////////////////////////////////////
	    pos_samples.convertTo( pos_samples, CV_32FC1 );
	    neg_samples.convertTo( neg_samples, CV_32FC1 );
	    ///////////////////////////////////////////////

	    GMM_Initialize_with_KMeans( gnome_norm, GMM_Mixtures_Number, pos_samples, GMM_Mixtures_Means_pos, GMM_Mixtures_Covs_pos, "POS", false );
	    GMM_Initialize_with_KMeans( gnome_norm, GMM_Mixtures_Number, neg_samples, GMM_Mixtures_Means_neg, GMM_Mixtures_Covs_neg, "NEG", false );

	    invert_3x3s( GMM_Mixtures_Covs_pos, GMM_Mixtures_Covs_pos_INV );
	    invert_3x3s( GMM_Mixtures_Covs_neg, GMM_Mixtures_Covs_neg_INV );
	    sqrt_det(    GMM_Mixtures_Covs_pos, GMM_Mixtures_Covs_pos_SQRT_DET );
	    sqrt_det(    GMM_Mixtures_Covs_neg, GMM_Mixtures_Covs_neg_SQRT_DET );

	    ///////////////////////////////////////////////
	    pos_samples.convertTo( pos_samples, CV_64FC1 );
	    neg_samples.convertTo( neg_samples, CV_64FC1 );
	    ///////////////////////////////////////////////

	    GMM_Mixtures_Weights_INITIALIZE( GMM_Mixtures_Number, GMM_Mixtures_Weights_pos ); // initialize before executing EM
	    GMM_Mixtures_Weights_INITIALIZE( GMM_Mixtures_Number, GMM_Mixtures_Weights_neg ); // initialize before executing EM
	    GMM_Mixtures_Weights_TEST(                            GMM_Mixtures_Weights_pos ); // test if weights sum up to 1
	    GMM_Mixtures_Weights_TEST(                            GMM_Mixtures_Weights_neg ); // test if weights sum up to 1

	    ////////////////////////////////////////////////////////
	    double likelihoodConstant = pow( (2*M_PI), 3/(float)2 ); // precompute constant needed for EM
	    ////////////////////////////////////////////////////////

	    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	                                                            std::cout << "Train Positive Samples" << std::endl;
	    GMM_Train(   pos_samples,
	                 GMM_Mixtures_Number,
	                 GMM_Mixtures_Means_pos,
	                 GMM_Mixtures_Covs_pos,
	                 GMM_Mixtures_Covs_pos_INV,
	                 GMM_Mixtures_Covs_pos_SQRT_DET,
	                 GMM_Mixtures_Weights_pos,
	                 likelihoodConstant,
	                 false );
	                                                            std::cout << "Train Negative Samples" << std::endl;
	    GMM_Train(   neg_samples,
	                 GMM_Mixtures_Number,
	                 GMM_Mixtures_Means_neg,
	                 GMM_Mixtures_Covs_neg,
	                 GMM_Mixtures_Covs_neg_INV,
	                 GMM_Mixtures_Covs_neg_SQRT_DET,
	                 GMM_Mixtures_Weights_neg,
	                 likelihoodConstant,
	                 false );


	    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	                                                            std::cout << "Calculate Probabilities" << std::endl;

	    cv::Mat probabilityMAP_pos = cv::Mat::zeros(pos_samples.rows+neg_samples.rows,1,CV_64FC1);
	  //cv::Mat probabilityMAP_neg = cv::Mat::zeros(pos_samples.rows+neg_samples.rows,1,CV_64FC1);

	    ///////////////////////////////////
	    int III = all_samples.rows;    ////
	    int KKK = GMM_Mixtures_Number; ////
	    ///////////////////////////////////
	    for     (int iii=0; iii<III; iii++)
	    {
	        double likelihood_pos = 0;
	        double likelihood_neg = 0;

	        for (int kkk=0; kkk<KKK; kkk++)
	        {
	               likelihood_pos += myLikelihood( GMM_Mixtures_Weights_pos.at<double>(kkk), all_samples.row(iii), GMM_Mixtures_Means_pos.row(kkk), GMM_Mixtures_Covs_pos_INV[kkk], GMM_Mixtures_Covs_pos_SQRT_DET[kkk], likelihoodConstant );
	               likelihood_neg += myLikelihood( GMM_Mixtures_Weights_neg.at<double>(kkk), all_samples.row(iii), GMM_Mixtures_Means_neg.row(kkk), GMM_Mixtures_Covs_neg_INV[kkk], GMM_Mixtures_Covs_neg_SQRT_DET[kkk], likelihoodConstant );
	        }

	        probabilityMAP_pos.at<double>(iii) = likelihood_pos / (likelihood_pos + likelihood_neg);
	    }
	    /////////////////////////////////////////////////////////////////////
	    /////////////////////////////////////////////////////////////////////
	    probabilityMAP_pos = probabilityMAP_pos.reshape( 1, gnome_INI.rows );
	  //probabilityMAP_neg = 1 - probabilityMAP_pos; ////////////////////////
	    /////////////////////////////////////////////////////////////////////
	    /////////////////////////////////////////////////////////////////////

	    cv::imshow("positive probability - Custom",probabilityMAP_pos);
	  //cv::imshow("negative probability - Custom",probabilityMAP_neg);
	    cv::waitKey();
	}

	void GMMs_withOpenCV( const cv::Mat& img, const cv::Mat& mask ){

	    cv::Mat all_samples, pos_samples, neg_samples;
	    std::cout << "Organize Samples" << std::endl;   
	    organizeSamples(img, mask, all_samples, pos_samples, neg_samples, true ); // results: Mx3,Nx3 // each row is a sample

	    // train the models
	    const cv::TermCriteria quick(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 10, FLT_EPSILON);
	    cv::EM pos_model(cv::EM::DEFAULT_NCLUSTERS, cv::EM::COV_MAT_GENERIC, quick);
	    cv::EM neg_model(cv::EM::DEFAULT_NCLUSTERS, cv::EM::COV_MAT_GENERIC, quick);

	    std::cout << "Train Positive Samples" << std::endl;   pos_model.train(pos_samples );
	    std::cout << "Train Negative Samples" << std::endl;   neg_model.train(neg_samples );

	  //GMMs_withOpenCV_printEMinfo( pos_model, neg_model );

	    // classify the image pixels
	    std::cout << "Calculate Probabilities" << std::endl;
	    ////////////////////////////////////////
	    cv::Mat neg_prob,  pos_prob;
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

	////////////////////////////////////////////////////////////////////////////////////////////
	// split the pixels of an image into positive and negative samples with respect to a mask //   fully functioning version // a bit slower but more readable
	////////////////////////////////////////////////////////////////////////////////////////////

	void organizeSamples( const cv::Mat& img, const cv::Mat& mask, cv::Mat& all_samples, cv::Mat& pos_samples, cv::Mat& neg_samples, bool withOpenCV){

	    CV_Assert(img.cols == mask.cols && img.rows == mask.rows && mask.type() == CV_64FC1);

	    std::vector<double> pos, neg, all;

	    const size_t  channels = img.channels();
	    double sample[channels];

	    for     (int yyy=0; yyy<img.rows; yyy++)
	    {   for (int xxx=0; xxx<img.cols; xxx++)
	        {
	            // 3 channel pixel to 1x3 mat

	            if (withOpenCV)
	            {
	                sample[0] = img.at<cv::Vec3b>(yyy,xxx)(0);
	                sample[1] = img.at<cv::Vec3b>(yyy,xxx)(1);
	                sample[2] = img.at<cv::Vec3b>(yyy,xxx)(2);
	            }
	            else
	            {
	                sample[0] = img.at<cv::Vec3f>(yyy,xxx)(0);
	                sample[1] = img.at<cv::Vec3f>(yyy,xxx)(1);
	                sample[2] = img.at<cv::Vec3f>(yyy,xxx)(2);
	            }

	            // appends 3 elements (of 3 channels)
	            // at the end of *pos* or *neg*
	            // look below for commented-out explanation with simpler code
	            if (mask.at<double>(yyy,xxx)>=0.5)   {   pos.insert( pos.end(), sample, sample+channels );   } // position_to_start_appending, source_first, source_last
	            else                                 {   neg.insert( neg.end(), sample, sample+channels );   }
	                                                     all.insert( all.end(), sample, sample+channels );
	        }
	    }
	    // std::vector -> cv::Mat           // 3Nx1 -> Nx3
	    pos_samples = cv::Mat(pos, true);   pos_samples = pos_samples.reshape( 1, pos.size()/channels ); // channels, rows
	    neg_samples = cv::Mat(neg, true);   neg_samples = neg_samples.reshape( 1, neg.size()/channels );
	    all_samples = cv::Mat(all, true);   all_samples = all_samples.reshape( 1, all.size()/channels );

	}

	/////////////////////////////////////////////////////////////////////////////
	// compute the log-likelihood of each pixel with with respect to the model //   fully functioning version // a bit slower but more readable
	/////////////////////////////////////////////////////////////////////////////
	void predictImage(const cv::Mat& img, const cv::EM&  model, cv::Mat& result){

	    result.create(img.size(), CV_64FC1);

	    cv::Mat_<double> sample(1, img.channels(), CV_64FC1); // 1x3 mat

	    for     (int yyy=0; yyy<img.rows; yyy++)
	    {   for (int xxx=0; xxx<img.cols; xxx++)
	        {
	            // 3 channel pixel to 1x3 mat
	            sample(0) = img.at<cv::Vec3b>(yyy,xxx)(0);
	            sample(1) = img.at<cv::Vec3b>(yyy,xxx)(1);
	            sample(2) = img.at<cv::Vec3b>(yyy,xxx)(2);

	            // returns a 2-element double vector, we use just
	            // the 1st element (likelihood logarithm value)
	            result.at<double>(yyy,xxx) = model.predict( sample )[0];
	        }
	    }
	}

	////////////////////////////
	// show a grayscale image //
	////////////////////////////
	void showGray( const cv::Mat& img, const std::string title, const int t ){

	    CV_Assert( img.channels() == 1 );

	    double               minVal,  maxVal;
	    cv::minMaxLoc( img, &minVal, &maxVal );

	    cv::Mat            temp;
	    img.convertTo(     temp, CV_32F, 1./(maxVal-minVal), -minVal/(maxVal-minVal));
	    cv::imshow( title, temp);
	    cv::waitKey(t);
	}

	~GMEM(){

		//nothing to do here

	}

};