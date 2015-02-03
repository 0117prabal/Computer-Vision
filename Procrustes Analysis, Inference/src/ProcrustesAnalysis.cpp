#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define UNIT_VARIANCE 10.0

using namespace std;
using namespace cv;


class ProcrustesAnalysis{

private:

    cv::Mat data;
    int m_iter, m_maxIter;
    float m_err, m_maxErr;
    int num_coords, num_sampls;

public:

	ProcrustesAnalysis(){
	    
	    m_iter=0;
	    m_maxIter=1000;
	    m_err = FLT_MAX;
	    m_maxErr = 1e-5;
	    num_coords=112;
	    num_sampls=40;

	}


	bool LoadData(string in_fpath){
	    
	    ifstream in(in_fpath.c_str());
	    if(!in) return false;

	    Mat indata(num_coords,num_sampls,CV_32F);
	    indata.setTo(0);
	    for(int row=0; row<indata.rows; ++row){
	        for(int col=0; col<indata.cols; ++col){
	            in >> indata.at<float>(row,col);
	        }
	    }
	    data = indata;
	    return true;
	}

	void AlignData(){
	    
	    Mat newdata;
	    while(m_iter<m_maxIter && m_err>=m_maxErr){
	        // align
	        FindAlignment(newdata);
	        // compute error
	        m_err = ComputeAvgError(newdata);
	        cout << m_iter << " " << m_err << endl;
	        // update
	        m_iter++;
	        data = newdata.clone();
	//        if(m_iter > 10) exit(-1);
	    }

	}


	void FindAlignment(Mat& out_newdata){
	    // find mean shape
	    Mat shape_mu;
	    ComputeMeanShape(data,shape_mu,UNIT_VARIANCE);

	    // align all shapes to mean
	    out_newdata = 0*data.clone();
	    for(int sidx=0; sidx<data.cols; ++sidx){
	        Mat out;
	        Align(data.col(sidx), shape_mu, out);
	        out.copyTo(out_newdata.col(sidx));
	    }
	}

	void ComputeMeanShape(Mat& in_data, Mat& out_mu, double in_ref_var){
	
	    Mat shape_mu, temp;
	    reduce(data,shape_mu,1,CV_REDUCE_AVG);

	    displayShape(shape_mu, "mean_shape", 1);

	    // get mean point
	    Point2f mean;
	    temp = shape_mu.clone();
	    temp = (temp.reshape(0,2)).t();
	    mean.x = sum(temp.col(0))[0]/temp.rows;
	    mean.y = sum(temp.col(1))[0]/temp.rows;

	    // get variance
	    double var_temp;
	    temp.col(0) = mean.x;
	    temp.col(1) = mean.y;
	    var_temp = sum(temp.mul(temp))[0];

	    // get scale
	    double scale = sqrt(in_ref_var/var_temp);
	    shape_mu *= scale;

	    out_mu = shape_mu.clone();
	    return;
	}

	void Align(Mat in_tgt, Mat& in_ref, Mat& out_aligned){

	    // separate out x and y
	    Mat tgt = in_tgt.clone();
	    Mat ref = in_ref.clone();
	    tgt = (tgt.reshape(0,2)).t();
	    ref = (ref.reshape(0,2)).t();

	    // get the means
	    Point2f mu_tgt, mu_ref;
	    mu_tgt.x = sum(tgt.col(0))[0]/tgt.rows;
	    mu_tgt.y = sum(tgt.col(1))[0]/tgt.rows;
	    mu_ref.x = sum(ref.col(0))[0]/ref.rows;
	    mu_ref.y = sum(ref.col(1))[0]/ref.rows;

	    // get the variance
	    double var_tgt, var_ref, scale;
	    tgt.col(0) -= mu_tgt.x;
	    tgt.col(1) -= mu_tgt.y;
	    ref.col(0) -= mu_ref.x;
	    ref.col(1) -= mu_ref.y;
	    Mat temp = tgt.clone();
	    var_tgt  = sum(temp.mul(temp))[0];
	    temp     = ref.clone();
	    var_ref  = sum(temp.mul(temp))[0];
	    scale = sqrt(var_ref/var_tgt);

	    // get the rotation matrix
	    tgt *= scale;
	    SVD svd;
	    Mat u,s,vt;
	    temp = tgt.t() * ref;
	    svd.compute(temp,s,u,vt);

	    // perform inverse transformation
	    Mat rot = vt.t() * u.t();
	    Mat out = (rot * tgt.t()).t();
	    out.col(0) += mu_ref.x;
	    out.col(1) += mu_ref.y;

	    // copy to output
	    out = out.t();
	    out = (out.reshape(0,1)).t();
	    out_aligned = out.clone();

    	return;
	}

	float ComputeAvgError(Mat& in_newdata){
	    
	    Mat temp;
	    ComputeMeanShape(in_newdata, temp, UNIT_VARIANCE);
	    temp = repeat(temp,1,in_newdata.cols);

	    temp -= (in_newdata);
	    temp = temp.mul(temp);
	    return sqrt(sum(temp)[0]/(data.cols*data.rows));
	}


	void displayShape(Mat& in_shapes, string header, int waitFlag){
	    
	    // init interm. parameters
	    int scl=500;
	    double maxval;
	    RNG rng;
	    Mat shapes = in_shapes.clone();
	    minMaxLoc(shapes,NULL,&maxval,NULL,NULL);
	    shapes *= (scl*0.8/maxval);

	    Mat dispImg(scl,scl,CV_8UC3);
	    Scalar color(0,0,0);
	    dispImg.setTo(color);
	    int lstx=shapes.rows/2-1;

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