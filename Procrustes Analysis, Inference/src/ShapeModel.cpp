#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define UNIT_VARIANCE 10.0

using namespace std;
using namespace cv;

class ShapeModel{

private:

    Mat meanShape;
    Mat prinComp;
    Mat prinVal;
    RNG rng;
    int scl;

public:

    ShapeModel(){rng(10); scl=400;};
    ~ShapeModel(){};

    /* variables */
    Mat trainD;
    Mat testD;


    void inference(){
    	
	    Mat tempTest, compWt, reconTest;
	    compWt.create(prinVal.rows,1,CV_32F);
	    testD.copyTo(tempTest);

	    // decomposition
	    tempTest -= meanShape;
	    for(int cidx=0; cidx<prinComp.cols; ++cidx){
	        float tempWt=0;
	        for(int ridx=0; ridx<prinComp.rows; ++ridx){
	            tempWt += prinComp.at<float>(ridx,cidx)*tempTest.at<float>(ridx,0);
	        }
	        compWt.at<float>(cidx,0) = tempWt;
	    }

	    // reconstruction
	    meanShape.copyTo(reconTest);
	    for(int cidx=0; cidx<prinComp.cols; ++cidx){
	        for(int ridx=0; ridx<prinComp.rows; ++ridx){
	            reconTest.at<float>(ridx,0) += prinComp.at<float>(ridx,cidx)*compWt.at<float>(cidx,0);
	        }
	    }

	    cout<<"Wt of mean = 1"<<endl;
	    for(int ridx=0; ridx<compWt.rows; ++ridx){
	        cout<<"Wt of PC"<<ridx<<"  = "<<compWt.at<float>(ridx,0)<<endl;
	    }

	    displayShape(testD,string("testSample"));
	    displayShape(reconTest,string("reconstructed test sample"));
	}

	void displayModel(){

	    for(int pidx=0; pidx<prinComp.cols; ++pidx){
	        for(int itr=0; itr<5; ++itr){
	            stringstream pcidx;
	            pcidx << pidx;
	            string title("Principle Component Variation: PC");
	            title += pcidx.str();

	            for(float dev=-0.3; dev<=0.3; dev+=0.1){
	                Mat tempShape;
	                float wgt = dev * sqrt(prinVal.at<float>(pidx,0));
	                meanShape.copyTo(tempShape);
	                for(int ridx=0; ridx<prinComp.rows; ++ridx){
	                    tempShape.at<float>(ridx,0) += wgt*prinComp.at<float>(ridx,pidx);
	                }
	                displayShape(tempShape,title,0);
	            }
	        }
	    }
	}

	void trainModel(){
	    // find mean
	    meanShape.create(trainD.rows,1,CV_32F);
	    meanShape.setTo(0);
	    float *dptr,*mptr,*diffPtr;
	    for(int ridx=0; ridx<trainD.rows; ++ridx){
	        dptr = trainD.ptr<float>(ridx);
	        mptr = meanShape.ptr<float>(ridx);
	        for(int cidx=0; cidx<trainD.cols; ++cidx,++dptr){
	            *mptr += *dptr;
	        }
	        *mptr /= trainD.cols;
	    }
	    displayShape(meanShape,string("meanShape"));

	    // find covariance
	    Mat diff(trainD.rows,trainD.cols,CV_32F);
	    diff.setTo(0);
	    for(int ridx=0; ridx<trainD.rows; ++ridx){
	        dptr    = trainD.ptr<float>(ridx);
	        diffPtr = diff.ptr<float>(ridx);
	        mptr    = meanShape.ptr<float>(ridx);
	        for(int cidx=0; cidx<trainD.cols; ++cidx,++dptr,++diffPtr){
	            *diffPtr = (*dptr)-(*mptr);
	        }
	    }
	    Mat diffT, covar;
	    transpose(diff,diffT);
	    covar = diff*diffT;

	    // find eigenvectors and eigen values
	    Mat eigval, eigvec, cumsum;
	    eigen(covar,eigval,eigvec);
	    transpose(eigvec,eigvec);
	    eigval.copyTo(cumsum);
	    for(int ridx=1; ridx<eigval.rows; ++ridx)
	        cumsum.at<float>(ridx,0) += cumsum.at<float>(ridx-1,0);
	    cumsum /= (sum(eigval))[0];

	    // store principle components
	    int numPrinComp=1;
	    for(int ridx=0; ridx<eigval.rows; ++ridx){
	        if(cumsum.at<float>(ridx,0)<0.90)
	            numPrinComp++;
	        else
	            break;
	    }
	    prinVal.create(numPrinComp,1,CV_32F);
	    for(int ridx=0; ridx<prinVal.rows; ++ridx)
	        prinVal.at<float>(ridx,0) = eigval.at<float>(ridx,0);
	    prinComp.create(eigvec.rows,numPrinComp,CV_32F);
	    for(int ridx=0; ridx<eigvec.rows; ++ridx){
	        dptr = eigvec.ptr<float>(ridx);
	        mptr = prinComp.ptr<float>(ridx);
	        for(int cidx=0; cidx<numPrinComp; ++cidx,++dptr,++mptr){
	            *mptr = *dptr;
	        }
	    }
	}

	void loadData(const string& fileLoc, Mat& data){

	    // check if file exists
	    ifstream iStream(fileLoc.c_str());
	    if(!iStream){
	        cout<<"file for load data cannot be found"<<endl;
	        exit(-1);
	    }

	    // read aligned hand shapes
	    int rows, cols;
	    iStream >> rows;
	    iStream >> cols;
	    data.create(rows,cols,CV_32F);
	    data.setTo(0);
	    float *dptr;
	    for(int ridx=0; ridx<data.rows; ++ridx){
	        dptr = data.ptr<float>(ridx);
	        for(int cidx=0; cidx<data.cols; ++cidx, ++dptr){
	            iStream >> *dptr;
	        }
	    }
	    iStream.close();
	}


	void displayShape(Mat& shapes,string header, int waitFlag = 1){
	    // init interm. parameters
	    Mat dispImg(scl,scl,CV_8UC3);
	    Scalar color(0,0,0);
	    dispImg.setTo(color);
	    int lstx = shapes.rows/2-1;

	    if(0==waitFlag){

	        color[0]=20; color[1]=40; color[2]=180;
	    }

	    // draw each input shape in a different color
	    for(int cidx=0; cidx<shapes.cols; ++cidx){
	        if(1==waitFlag){
	            color[0]=rng.uniform(0,256); color[1]=rng.uniform(0,256); color[2]=rng.uniform(0,256);
	        }
	        for(int ridx=0; ridx<lstx-1; ++ridx){
	            Point2i startPt(shapes.at<float>(ridx,cidx),shapes.at<float>(ridx+lstx+1,cidx));
	            Point2i endPt(shapes.at<float>(ridx+1,cidx),shapes.at<float>(ridx+lstx+2,cidx));
	            line(dispImg,startPt,endPt,color,2);
	        }
	        imshow(header.c_str(),dispImg);
	        waitKey(150);
	    }
	    if(1==waitFlag){
	        cout<<"press any key to continue..."<<endl;
	        waitKey(0);
	    }
	}

};