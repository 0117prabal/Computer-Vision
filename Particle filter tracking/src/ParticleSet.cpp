#ifndef PARTICLESET_CPP
#define PARTICLESET_CPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>


using namespace std;
using namespace cv;

class ParticleSet{

public:
    
    std::vector<Particle> particles;
    double FrameRangeX;
    double FrameRangeY;
    double maxWeight_BestFit;
    int    maxWeight_BestFit_ParticleID;
    Particle detection;

    void find_ROIs_forParticles( cv::Mat &currFrame )
	{
	    for (size_t iii=0; iii<particles.size(); iii++)
	    {
	        cv::Rect  rectt(particles[iii].xxx,particles[iii].yyy, particles[iii].www, particles[iii].hhh);
	        currFrame(rectt).copyTo( particles[iii].roi );
	    }
	}


	void show_ROIs_forParticles_onFrame( cv::Mat &currFrameSHOW, bool shouldImShow )
	{
	    for (size_t iii=0; iii<particles.size(); iii++)
	    {
	        cv::Rect                    rectt(particles[iii].xxx,particles[iii].yyy,particles[iii].www,particles[iii].hhh);
	        cv::rectangle(currFrameSHOW,rectt,cv::Scalar(0,0,255));
	    }
	    if (shouldImShow)   cv::imshow("Particles",currFrameSHOW);
	}

	void calc_hist_forParticles()
	{
	    for (size_t iii=0; iii<particles.size(); iii++)     particles[iii].calculateHistogramm();
	}

	void calc_distWeights_particles2detection() // 1==PerfectMatch // 0==Irrelevant
	{
	    for (size_t iii=0; iii<particles.size(); iii++)
	    {
	        double bhDist = cv::compareHist( particles[iii].hist, detection.hist, CV_COMP_BHATTACHARYYA );
	      //particles[iii].weight =     1 - bhDist  ;
	        particles[iii].weight =   exp( -bhDist / (2*0.080) ); // better!!!
	    }
	}


	// Explanation for Resampling with Replacement with a Resampling Wheel
	// https://www.youtube.com/watch?v=wNQVo6uOgYA#t=0
	// Works in Roulette-like way. Instead of a static pointer and rotating roulette,
	// there's a static roulette and rotating pointer. The arc of each index is prop. to its likelihood
	// An alternative approach would be to use a vector-approach as described in the slides
	int particleID_basedOnArc360( double arcIN )
	{
	    for (size_t iii=1; iii<particles.size(); iii++)
	    {
	        if (arcIN < particles[iii].arcCircleStart360)   return iii-1;
	    }
	    return particles.size()-1;
	}
	void arrangeWeightsInCircle()
	{
	    double summm = 0;
	    for (size_t iii=0; iii<particles.size(); iii++)
	    {
	        particles[iii].arcCircleStart360 = summm * 360;
	        summm += particles[iii].weightNormalized;
	    }
	}

	void calculateMaxWeight()
	{
	    maxWeight_BestFit            = -888; // reset to 'default' dummy value
	    maxWeight_BestFit_ParticleID = -888; // reset to 'default' dummy value

	    for (size_t iii=0; iii<particles.size(); iii++)
	    {
	        if (particles[iii].weight>maxWeight_BestFit)
	        {
	            maxWeight_BestFit            = particles[iii].weight;
	            maxWeight_BestFit_ParticleID = iii;
	        }
	    }
	}
	void calcNormalizedWeights()
	{
	    double summm = 0;       for (size_t iii=0; iii<particles.size(); iii++)    summm += particles[iii].weight;
	                            for (size_t iii=0; iii<particles.size(); iii++)    particles[iii].weightNormalized = particles[iii].weight / summm;
	}



	void createSet( int particleNumber )
	{
	    srand(time(NULL)); // seed for random number generation

	                                                    //////////////////////////////////////////////////////////////////////////////////
	    for (int iii=0; iii<particleNumber; iii++)      particles.push_back(   generateNewParticleAccordingToExistingOne( detection )   );
	                                                    //////////////////////////////////////////////////////////////////////////////////

	    this->maxWeight_BestFit = -888; // reset to 'default' dummy value
	}


	Particle generateNewParticleAccordingToExistingOne( Particle &existingParticle )
	{
	    // used for motion/noise model
	    int varXXX;
	    int varYYY;

	    // used for security check
	    cv::Rect  frameRect(0,0,FrameRangeX,FrameRangeY);
	    cv::Point topLeft;
	    cv::Point bottomRight;

	    // insert Gaussian Noise, iterate as long as it gets out of bounds
	    // number of particles remains the same after resampling
	    ////////////////////////////////////////////////////////////////////////////////////////////////
	    ////////////////////////////////////////////////////////////////////////////////////////////////
	    do
	    {
	        varXXX = (int)round( myRandomNumber_sign()*myGaussian(0.0,0.10,myRandomNumber_0_1()) * 40 );
	        varYYY = (int)round( myRandomNumber_sign()*myGaussian(0.0,0.10,myRandomNumber_0_1()) * 40 );

	        // test - out of image bound OR non-clockwise rectangle
	        topLeft.x     = existingParticle.xxx+varXXX;
	        topLeft.y     = existingParticle.yyy+varYYY;
	        bottomRight.x = existingParticle.xxx+varXXX + existingParticle.www;
	        bottomRight.y = existingParticle.yyy+varYYY + existingParticle.hhh;

	    }///////////////////////////////////////////////////////////////////////////////////////////////
	    while ( frameRect.contains( topLeft     ) == false ||
	            frameRect.contains( bottomRight ) == false ||
	            bottomRight.x <= topLeft.x                 ||
	            bottomRight.y <= topLeft.y                 );
	    ////////////////////////////////////////////////////////////////////////////////////////////////
	    ////////////////////////////////////////////////////////////////////////////////////////////////

	    // generation of new particle
	    Particle newParticle;
	             newParticle.xxx = existingParticle.xxx + varXXX; // actually introduce noise
	             newParticle.yyy = existingParticle.yyy + varYYY; // after the security check
	             newParticle.www = existingParticle.www;
	             newParticle.hhh = existingParticle.hhh;
	             newParticle.weight            = -888; // reset to 'default' dummy value
	             newParticle.weightNormalized  = -888; // reset to 'default' dummy value
	             newParticle.arcCircleStart360 = -888; // reset to 'default' dummy value
	    return   newParticle;

	}


	void resampling_andReplaceParticles()
	{

	        // temp structure for new particles
	        std::vector<Particle> newParticles;

	        int currStart = (int)round( myRandomNumber_Range( particles.size()-1 ) ); // random initial particleID

	        for (size_t iii=0; iii<particles.size(); iii++)
	        {
	                // Resampling wheel, see link included at the corresponding methods
	                double currAdd = myRandomNumber_Range( 2*maxWeight_BestFit ) * 360;
	                currStart = fmod( currStart+currAdd, 360 ); // modulo operator for floats // operator '%' defined only for integers
	                int particleID = particleID_basedOnArc360(currStart);

	                /////////////////////////////////////////////////////////////////////////////////////////////////
	                newParticles.push_back(   generateNewParticleAccordingToExistingOne( particles[particleID] )   );
	                /////////////////////////////////////////////////////////////////////////////////////////////////
	        }

	        // Replace old with new particles !!!
	        // Number of particles remains the same after resampling
	        for (size_t iii=0; iii<particles.size(); iii++)     particles[iii] = newParticles[iii];

	}


	void findDetection_BestParticle()
	{
	    detection.xxx    = particles[maxWeight_BestFit_ParticleID].xxx;
	    detection.yyy    = particles[maxWeight_BestFit_ParticleID].yyy;
	    detection.www    = particles[maxWeight_BestFit_ParticleID].www;
	    detection.hhh    = particles[maxWeight_BestFit_ParticleID].hhh;
	    detection.weight = particles[maxWeight_BestFit_ParticleID].weight;
	    detection.roi    = particles[maxWeight_BestFit_ParticleID].roi.clone();
	  //detection.hist   = particles[maxWeight_BestFit_ParticleID].hist.clone(); // do NOT copy // keep initial histogram
	}


	void findAverageParticle()
	{
	    double meanXXX = 0;
	    double meanYYY = 0;
	    int    counter = 0;

	    for (size_t iii=0; iii<particles.size(); iii++)
	    {
	        // The weights are already normalized (sum = 1).
	        // For the final "detection" we take the weighted average of all particles.
	        // One can also take the top x% of tha particles, or the particles with weight above x%*max_weight.
	        // Just the max value works, but is more noisy (follows the object with more jitter)
	        meanXXX += static_cast<double>(particles[iii].xxx) * particles[iii].weightNormalized;
	        meanYYY += static_cast<double>(particles[iii].yyy) * particles[iii].weightNormalized;
	        counter++;
	    }
	    meanXXX = meanXXX;
	    meanYYY = meanYYY;

	    detection.xxx = static_cast<int>(round(meanXXX));
	    detection.yyy = static_cast<int>(round(meanYYY));
	    detection.www = particles[0].www;
	    detection.hhh = particles[0].hhh;
	}

	double myRandomNumber_sign(){   
		
		if (myRandomNumber_0_1()<0.5)
			return -1;
        else
        	return +1;

    }

    double myGaussian( double mu, double sigma2, double xxx )       {       return (   exp( -0.5 * pow(xxx-mu,2) / sigma2  )   /   sqrt(2*M_PI*sigma2)   );     }

	double myRandomNumber_0_1()                    {   return  static_cast<double>(rand())/static_cast<double>(RAND_MAX);  }
	double myRandomNumber_Range( double range )    {   return  myRandomNumber_0_1() * range;}

};

#endif