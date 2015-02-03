#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ProcrustesAnalysis.cpp"
#include "ShapeModel.cpp"

using namespace std;
using namespace cv;

#define UNIT_VARIANCE 10.0

int main(){

	/*ProcrustesAnalysis proc;
    proc.LoadData(string("images/hands_orig_train.txt"));
    proc.AlignData();*/

    string trainLoc = "images/hands_aligned_train.txt";//argv[1];
    ShapeModel model;
    model.loadData(trainLoc,model.trainD);
    model.displayShape(model.trainD,string("trainingShapes"));

    model.trainModel();
    model.displayModel();

    string testLoc = "images/hands_aligned_test.txt"; //argv[2];
    model.loadData(testLoc,model.testD);
    model.inference();

	return 0;
}