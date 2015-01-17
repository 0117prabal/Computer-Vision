#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;


class GMEM{

private:

	Mat image;
	int    PARAM_Max_Iter_Numb    = 1;
	double PARAM_Epsilon_converge = 5.0;
	double PARAM_Epsilon_assert   = FLT_EPSILON;
	int    GMM_Mixtures_Number    = 10;

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

	    //GMMs_withOpenCV(image, mask);

	    GMMs_Custom(image, mask);
	}

	void GMMs_Custom( const cv::Mat& gnome_INI, const cv::Mat& mask ){

	    cv::Mat gnome_norm;
	    gnome_INI.convertTo(   gnome_norm, CV_32FC3, (1.0/255.0) ); // image normalized [0,255] -> [0,1]
	    //gnome_norm.reshape( 1, gnome_norm.total() );

	    cv::Mat all_samples, pos_samples, neg_samples;
	    std::cout << "Organize Samples" << std::endl;
	    organizeSamples( gnome_norm, mask, all_samples, pos_samples, neg_samples, false ); // results: Mx3,Nx3 // each row is a sample
	    

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
	    sqrt_det(GMM_Mixtures_Covs_pos, GMM_Mixtures_Covs_pos_SQRT_DET );
	    sqrt_det(GMM_Mixtures_Covs_neg, GMM_Mixtures_Covs_neg_SQRT_DET );

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


	void GMM_Initialize_with_KMeans( const cv::Mat& img_norm, const int GMM_Mixtures_Number, const cv::Mat& GMM_Samples_Val, cv::Mat& GMM_Mixtures_Means, std::vector< cv::Mat>&   GMM_Mixtures_Covs, const std::string& title, const bool shouldPrint ){

	    if (shouldPrint)
	    {
	        std::cout << "/////////////////////////////////////////////////////////////////////////"         << std::endl;
	        std::cout << "////////////////////////////////// "<<title<<" //////////////////////////////////" << std::endl;
	        std::cout << "/////////////////////////////////////////////////////////////////////////"         << std::endl;
	    }


	    cv::Mat GMM_Samples_MixtureID;
	    cv::kmeans(GMM_Samples_Val, GMM_Mixtures_Number, GMM_Samples_MixtureID, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.001), 10, cv::KMEANS_PP_CENTERS, GMM_Mixtures_Means);


	    cv::Mat GMM_Samples_Means = cv::Mat::zeros( GMM_Samples_MixtureID.rows, GMM_Samples_MixtureID.cols, CV_32FC3 );
	    for (int iii=0; iii<GMM_Samples_Val.rows; iii++){

	    	memcpy(GMM_Samples_Means.ptr(iii), GMM_Mixtures_Means.ptr(*GMM_Samples_MixtureID.ptr(iii)), GMM_Samples_Means.elemSize()*GMM_Samples_MixtureID.cols); // destination // source

		}

	    GMM_Samples_Means. convertTo(GMM_Samples_Means,  CV_64FC3);
	    GMM_Mixtures_Means.convertTo(GMM_Mixtures_Means, CV_64FC1);
		GMM_Mixtures_Covs.resize( GMM_Mixtures_Number);

	    for (int nnn=0; nnn < GMM_Mixtures_Number; nnn++){

	      GMM_Mixtures_Covs[nnn] = cv::Mat::eye(3,3,CV_64FC1);

	  	}

	    std::vector< int > GMM_Mixtures_Covs_count;
	                       GMM_Mixtures_Covs_count.resize( GMM_Mixtures_Number );
	            std::fill( GMM_Mixtures_Covs_count.begin(),
	                       GMM_Mixtures_Covs_count.end(),0);

	    /////////////////////////////////////////////////////////////////////////
	    //SIGMA = E[(X1-mu1)(X1-mu1)] E[(X1-mu1)(X2-mu2)] ... E[(X1-mu1)(X*-mu*)]   in our case we will use only the
	    //        E[(X2-mu2)(X1-mu1)] E[(X2-mu2)(X2-mu2)] ... E[(X2-mu2)(X*-mu*)]   diagonal form of the covariance matrix
	    //        E[(X*-mu*)(X1-mu1)] E[(X*-mu*)(X2-mu2)] ... E[(X*-mu*)(X*-mu*)]
	    /////////////////////////////////////////////////////////////////////////
	    for (int iii=0; iii<GMM_Samples_Val.rows; iii++)
	    {
	        int  mix = GMM_Samples_MixtureID.at<int>(iii);

	        GMM_Mixtures_Covs_count[mix]++;

	        GMM_Mixtures_Covs[mix].at<double>(0,0) += pow( GMM_Samples_Val.at<double>(iii,0) - GMM_Mixtures_Means.at<double>(mix,0), 2 );
	        GMM_Mixtures_Covs[mix].at<double>(1,1) += pow( GMM_Samples_Val.at<double>(iii,1) - GMM_Mixtures_Means.at<double>(mix,1), 2 );
	        GMM_Mixtures_Covs[mix].at<double>(2,2) += pow( GMM_Samples_Val.at<double>(iii,2) - GMM_Mixtures_Means.at<double>(mix,2), 2 );
	    }
	    for (int nnn=0; nnn<GMM_Mixtures_Number; nnn++)
	        GMM_Mixtures_Covs[nnn] /= (double)GMM_Mixtures_Covs_count[nnn];

	    if (shouldPrint)
	    {
	                                                                std::cout << "*****************************************************"                   << std::endl;
	            for (int nnn=0; nnn<GMM_Mixtures_Number; nnn++)     std::cout << GMM_Mixtures_Covs[nnn] << "\t\t" << GMM_Mixtures_Covs_count[nnn] << "\n"  << std::endl;
	                                                                std::cout << "*****************************************************"                   << std::endl;



	            std::cout << "\n" << "GMM_Initialize_with_KMeans" << std::endl;
	            for     (int iii=0; iii<GMM_Mixtures_Means.rows; iii++) { std::cout << iii+1;
	                for (int jjj=0; jjj<GMM_Mixtures_Means.cols; jjj++)
	                     std::cout  << "\t" << GMM_Mixtures_Means.at<double>(iii,jjj);
	                     std::cout  <<  std::endl;
	            }
	            std::cout <<                                                                                                                                         std::endl;
	            std::cout << "GMM_Samples_MixtureID     " << GMM_Samples_MixtureID.rows<<"x"<<GMM_Samples_MixtureID.cols                                          << std::endl;
	            std::cout << "GMM_Samples_Val           " << GMM_Samples_Val.      rows<<"x"<<GMM_Samples_Val.      cols << " ~ " << GMM_Samples_Val.  channels() << std::endl;
	            std::cout << "GMM_Mixtures_Means        " << GMM_Mixtures_Means.   rows<<"x"<<GMM_Mixtures_Means.   cols <<                                          std::endl;
	            std::cout << "GMM_Mixtures_Covs         " << GMM_Mixtures_Covs.    size()                                                                         << std::endl;
	            std::cout << "GMM_Mixtures_Covs[iii]    " << GMM_Mixtures_Covs[0]. rows<<"x"<<GMM_Mixtures_Covs[0]. cols                                          << std::endl;
	            std::cout << "img_norm                  " << img_norm.             rows<<"x"<<img_norm.             cols                                          << std::endl;
	            std::cout <<                                                                                                                                         std::endl;
	    }
	}

	void invert_3x3s( const std::vector< cv::Mat >& input_3x3s, std::vector< cv::Mat >& output_3x3s ){

	    output_3x3s.resize( input_3x3s.size() );
	    
	    for (size_t iii=0; iii<input_3x3s.size(); iii++){

	    	output_3x3s[iii]  = input_3x3s[iii].inv();

	    }
	}

	void sqrt_det(const std::vector< cv::Mat >& input_3x3s, std::vector< double  >& output_doubles ){

	    output_doubles.resize(input_3x3s.size());
	    for (size_t iii=0; iii<input_3x3s.size(); iii++){

	    	output_doubles[iii] = sqrt( cv::determinant( input_3x3s[iii] ) );

	    }

	}

	void GMM_Mixtures_Weights_INITIALIZE( const int GMM_Mixtures_Number, cv::Mat& GMM_Mixtures_Weights ){

	    GMM_Mixtures_Weights = cv::Mat( GMM_Mixtures_Number, 1, CV_64FC1 );

	    double value_normal = 1/(double)GMM_Mixtures_Number;

	    for (int nnn=0; nnn<GMM_Mixtures_Weights.rows; nnn++)
	    {
	        GMM_Mixtures_Weights.at<double>(nnn) = value_normal;
	    }

	}


	void GMM_Mixtures_Weights_TEST( const cv::Mat& GMM_Mixtures_Weights ){

	    if( !(fabs( cv::sum(GMM_Mixtures_Weights)[0] - 1 ) <= PARAM_Epsilon_assert) )
	    std::cout << "GMM_Mixtures_Weights_TEST" << "\t\t" <<  fabs( cv::sum(GMM_Mixtures_Weights)[0] - 1 ) << "\t\t" << PARAM_Epsilon_assert << std::endl;
	    
	    assert(fabs( cv::sum(GMM_Mixtures_Weights)[0] - 1 ) <= PARAM_Epsilon_assert);
	}

	void GMM_Train(   	const cv::Mat&            samples,
                  		const int&                GMM_Mixtures_Number,
                        cv::Mat&                  GMM_Mixtures_Means,
                        std::vector< cv::Mat >&   GMM_Mixtures_Covs,
                        std::vector< cv::Mat >&   GMM_Mixtures_Covs_INV,
                        std::vector< double  >&   GMM_Mixtures_Covs_SQRT_DET,
                        cv::Mat&                  GMM_Mixtures_Weights,
                  		const double              likelihoodConstant,
                  		const bool                shouldPrint)
	{

	    /////////////////////////////////////////////
	    /////////////////////////////////////////////
	    int    ITER_COUNT                        = 0;
	    double GMM_Samples_MixtureMAXVAL_SUM_OLD = 0;
	    /////////////////////////////////////////////
	    /////////////////////////////////////////////


	    while(true)
	    {

	            /////////////
	            ITER_COUNT++;
	            /////////////


	            if (shouldPrint)
	            {
	                std::cout <<                                                                                                                                                         std::endl;
	                std::cout << "**************************************************************************************************************************" <<                         std::endl;
	                std::cout << "**************************************************************************************************************************" << "\t\t" << ITER_COUNT << std::endl;
	                std::cout << "**************************************************************************************************************************" <<                         std::endl;
	                std::cout <<                                                                                                                                                         std::endl;
	            }


	            //////////////////////////////
	            //////////////////////////////
	            int III = samples.rows;
	            int KKK = GMM_Mixtures_Number;
	            //////////////////////////////
	            //////////////////////////////


	            /////////////////////////////////////////////////////////////
	            /////////////////////////////////////////////////////////////
	            invert_3x3s( GMM_Mixtures_Covs, GMM_Mixtures_Covs_INV      );
	            sqrt_det(    GMM_Mixtures_Covs, GMM_Mixtures_Covs_SQRT_DET );
	            /////////////////////////////////////////////////////////////
	            /////////////////////////////////////////////////////////////
	            cv::Mat                                                                    GMM_Mixtures_Responsibility;                                           // CV_64FC1
	            GMM_Mixtures_Responsibility_INITIALIZE( samples.rows, GMM_Mixtures_Number, GMM_Mixtures_Responsibility ); // initialize before executing EM



	            if (shouldPrint)
	            {
	                std::cout <<                                          std::endl;
	                std::cout << "###################################" << std::endl;
	                std::cout << "###################### E-Step #####" << std::endl;
	                std::cout << "###################################" << std::endl;
	                std::cout <<                                          std::endl;
	            }
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            // formula 7.17, page 110, Prince book /////////////////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            for     (int iii=0; iii<III; iii++)
	            {   for (int kkk=0; kkk<KKK; kkk++)
	                {
	                    double                              denomin  = 0;
	                    for (int jjj=0; jjj<KKK; jjj++)     denomin += myLikelihood( GMM_Mixtures_Weights.at<double>(jjj), samples.row(iii), GMM_Mixtures_Means.row(jjj), GMM_Mixtures_Covs_INV[jjj], GMM_Mixtures_Covs_SQRT_DET[jjj], likelihoodConstant );
	                    double                                nomin  = myLikelihood( GMM_Mixtures_Weights.at<double>(kkk), samples.row(iii), GMM_Mixtures_Means.row(kkk), GMM_Mixtures_Covs_INV[kkk], GMM_Mixtures_Covs_SQRT_DET[kkk], likelihoodConstant );

	                    GMM_Mixtures_Responsibility.at<double>(iii,kkk) = nomin / denomin;                                                                              // GMM_Mixtures_Responsibility
	                }
	            }
	            ////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////
	            GMM_Mixtures_Responsibility_TEST( GMM_Mixtures_Responsibility );
	            ////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////





	            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            //////////    TERMINATION TESTS    ///////////////////////////////////////////////////////////////////////////////////////////////////
	            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            cv::Mat                                   GMM_Samples_MixtureMAXVAL;
	            cv::reduce( GMM_Mixtures_Responsibility,  GMM_Samples_MixtureMAXVAL, 1, CV_REDUCE_MAX );
	            double           maxElementSum =  cv::sum(GMM_Samples_MixtureMAXVAL)[0];
	            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            double                              GMM_Samples_MixtureMAXVAL_SUM  = maxElementSum;
	            for (int iii=0; iii<III; iii++)     GMM_Samples_MixtureMAXVAL_SUM += cv::sum(GMM_Mixtures_Responsibility.row(iii))[0] / maxElementSum;
	            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            if (ITER_COUNT > PARAM_Max_Iter_Numb)                                   break;
	            //////////////////////////////////////////////////////////////////////////////
	            if (shouldPrint)
	            std::cout << fabs( GMM_Samples_MixtureMAXVAL_SUM -
	                               GMM_Samples_MixtureMAXVAL_SUM_OLD ) << std::endl;
	            ////////////////////////////////////////////////////////////////////
	            if (fabs(GMM_Samples_MixtureMAXVAL_SUM  -
	                     GMM_Samples_MixtureMAXVAL_SUM_OLD) < PARAM_Epsilon_converge)   break;
	            //////////////////////////////////////////////////////////////////////////////
	            GMM_Samples_MixtureMAXVAL_SUM_OLD = GMM_Samples_MixtureMAXVAL_SUM;
	            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




	            if (shouldPrint)
	            {
	                std::cout <<                                          std::endl;
	                std::cout << "###################################" << std::endl;
	                std::cout << "###################### M-Step #####" << std::endl;
	                std::cout << "###################################" << std::endl;
	                std::cout <<                                          std::endl;
	            }
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            // formulas 7.19, page 113, Prince book ////////////////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            std::vector< double >   rik_sum_I;
	                                    rik_sum_I.resize( KKK );
	                       std::fill(   rik_sum_I.begin(),
	                                    rik_sum_I.end(), 0  );
	            double                  rik_sum_IK   =   0;

	            for (int kkk=0; kkk<KKK; kkk++)
	            {
	                double            add = cv::sum(GMM_Mixtures_Responsibility.col(kkk))[0];
	                rik_sum_I[kkk] += add;                                                                                                                              // rik_sum_I[kkk]
	                rik_sum_IK     += add;                                                                                                                              // rik_sum_IK
	            }
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            for (int kkk=0; kkk<KKK; kkk++)
	            {
	                GMM_Mixtures_Weights.row(kkk) = rik_sum_I[kkk] / rik_sum_IK;                                                                                        // GMM_Mixtures_Weights.row(kkk)
	            }
	                GMM_Mixtures_Weights_TEST( GMM_Mixtures_Weights );
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	                    GMM_Mixtures_Means.setTo(0);

	            for     (int kkk=0; kkk<KKK; kkk++)
	            {   for (int iii=0; iii<III; iii++)
	                {
	                    GMM_Mixtures_Means.row(kkk) += GMM_Mixtures_Responsibility.at<double>(iii,kkk) * samples.row(iii);
	                }   GMM_Mixtures_Means.row(kkk) /= rik_sum_I[kkk];                                                                                                  // GMM_Mixtures_Means.row(kkk)
	            }
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	                std::fill(  GMM_Mixtures_Covs.begin(),
	                            GMM_Mixtures_Covs.end(), cv::Mat::zeros(3,3,CV_64FC1)  );

	            for     (int kkk=0; kkk<KKK; kkk++)
	            {   for (int iii=0; iii<III; iii++)
	                {
	                    cv::Mat GMM_Mixture_Means_kkk = GMM_Mixtures_Means.row(kkk).t();   //  3x1

	                    cv::Mat samples_iii = samples.row(iii).t();                        //  3x1

	                    GMM_Mixtures_Covs[kkk] +=   GMM_Mixtures_Responsibility.at<double>(iii,kkk) *
	                                              ( samples_iii - GMM_Mixture_Means_kkk )           *
	                                              ( samples_iii - GMM_Mixture_Means_kkk ).t();
	                }
	                GMM_Mixtures_Covs[kkk] = GMM_Mixtures_Covs[kkk] / rik_sum_I[kkk];                                                                                  // GMM_Mixtures_Covs[kkk]
	            }
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	    }

	}

	~GMEM(){

		//nothing to do here

	}

};