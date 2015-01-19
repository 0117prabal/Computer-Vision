#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include "Particle.cpp"
#include "ParticleSet.cpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv){

	        /////////////////////
        int totalFrames = 32;
        ///////////////////////////////////
        bool shouldImShow_Detecions = true;
        bool shouldImShow_Particles = true;
        /////////////////////////////////////////////
        std::string stringBasePATH  = "images/";
        std::string stringExtension = ".png";
        /////////////////////////////////////////////

        std::cout << std::endl << std::endl;

        ParticleSet particleSet;
                    //////////////////////////////
                    particleSet.FrameRangeX = 720;
                    particleSet.FrameRangeY = 480;
                    //////////////////////////////


        for (int iii=0; iii<totalFrames; iii++)
        {
                // Read Current Frame
                std::ostringstream              stringStream;
                                                stringStream << stringBasePATH << iii+1 << stringExtension;
                cv::Mat currFrame = cv::imread( stringStream.str() );
                std::cout << stringStream.str () << std::endl;
                if (currFrame.empty())   std::cout << std::endl << std::endl << "main - Empty Image - Not read from disk" << std::endl << std::endl;



                // If first Frame, then create particle set around manual initialization
                if (iii==0)
                {
                        std::cout << "frame " << iii+1 << " / " << totalFrames << "   \t" << "Detection Manually Annotated" << std::endl;

                        particleSet.detection.initializeManually( currFrame, 448,191, 38,33 );
                        particleSet.detection.weight = -888; // reset to 'default' dummy value
                        cv::imwrite("nemoInitial.png",particleSet.detection.roi);
                        particleSet.createSet( 100 ); // number of particles (this number will be constant - resampling with substitution is used)
                        /////////
                        continue;
                        /////////
                }

                particleSet.resampling_andReplaceParticles(); // resample (preparations done at previous iteration)

                particleSet.find_ROIs_forParticles( currFrame );
                particleSet.calc_hist_forParticles();
                particleSet.calc_distWeights_particles2detection(); // max Weight is BeSt

                // find best particle // TODO-average
                particleSet.calcNormalizedWeights();
                particleSet.calculateMaxWeight();

              //particleSet.findDetection_BestParticle(); // uncomment this line and comment the next one, if you want to try just the strongest particle
                particleSet.findAverageParticle();

                particleSet.arrangeWeightsInCircle(); // preparation for resampling

                // Visualization & Write to disk
                // Tracked state @ current frame
                std::ostringstream       stringStreamSHOW;
                                         stringStreamSHOW << "DETECTION_" << iii+1 << ".png";
                cv::Mat  currFrameSHOW = currFrame.clone();
                particleSet.detection.show_onFrame( currFrameSHOW, shouldImShow_Detecions );
#if 0
                cv::imwrite( stringStreamSHOW.str(),currFrameSHOW  );
#endif

                // Visualization & Write to disk
                // Tracked state @ current frame and all particles
                cv::Mat  currFrameSHOW2 = currFrame.clone();
                std::ostringstream        stringStreamSHOW2;
                                          stringStreamSHOW2 << "PARTICLES_" << iii+1 << ".png";
                particleSet.show_ROIs_forParticles_onFrame( currFrameSHOW2, shouldImShow_Particles );
#if 0
                cv::imwrite(    stringStreamSHOW2.str(),     currFrameSHOW2  );
#endif

                std::cout << "frame " << iii+1 << " / " << totalFrames << "  \t" << "BestFit = " << particleSet.maxWeight_BestFit << "\t  particle ID = " << particleSet.maxWeight_BestFit_ParticleID << std::endl;

                if (shouldImShow_Detecions || shouldImShow_Particles)   cv::waitKey();
        }

        std::cout << std::endl << std::endl;
        cv::waitKey();


}