#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>
#include "GaussianPyramids.cpp"
#include "LaplacianPyramids.cpp"
#include "OrientedPyramids.cpp"
#include "CSPyramids.cpp"
#include "SaliencyMap.cpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv){

	if(argc == 1){

		cout << "usage: Please give the image path as command line input to generate the Saliency Map!" << endl;
		exit(1);
	}

	Mat image = imread(argv[1], 1);

	imshow("image", image);
	waitKey(0);
	
	// generate the gaussian pyramids
	GaussianPyramids pyramids(image, 5);
	pyramids.build();

	//generate the laplacian pyramids
	LaplacianPyramids lpyramids;
	lpyramids.build(pyramids.getLpyramid(), pyramids.getApyramid(), pyramids.getBpyramid());

	//generate the CS and SC pyramids for L channel
	CSPyramids csLpyramids;
	csLpyramids.build(pyramids.getLpyramid());

	//generate the CS and SC pyramids for A channel
	CSPyramids csApyramids;
	csApyramids.build(pyramids.getApyramid());

	//generate the CS and SC pyramids for B channel
	CSPyramids csBpyramids;
	csBpyramids.build(pyramids.getBpyramid());

	//generate the oriented pyramids for laplacian of L channel
	OrientedPyramids opyramidL;
	opyramidL.build(lpyramids.getLLpyramid());

	//generate the oriented pyramids for laplacian of A channel
	OrientedPyramids opyramidA;
	opyramidA.build(lpyramids.getLApyramid());

	//generate the oriented pyramids for laplacian of B channel
	OrientedPyramids opyramidB;
	opyramidB.build(lpyramids.getLBpyramid());

	//see the respective functions for each implementation
	SaliencyMap map;
	map.setFeatureMapO(opyramidL.getPyramid0(), opyramidL.getPyramid45(), opyramidL.getPyramid90(), opyramidL.getPyramid135());
	map.setFeatureMapI(csLpyramids.getCS(), csLpyramids.getSC());
	map.setFeatureMapCA(csApyramids.getCS(), csApyramids.getSC());
	map.setFeatureMapCB(csBpyramids.getCS(), csBpyramids.getSC());
	map.setConspicuityMaps();
	map.showSaliencyMap(image.size());

	return 0;
}