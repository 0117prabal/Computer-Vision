#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class Snakes{

  private:

      Mat image;

  public:

  Snakes(string path){

    image = imread(path, 1);

    if(!image.data){

        cout<<"Unable to read data"<<endl;
    }

  }

  void run(){

    vector<cv::Point2i> snake;
    Point2i center = Point2i(image.cols/2, image.rows/2);
    int radius = min(image.cols/3, image.rows/3);
    evolveSnakes(center, radius, snake);

  }

  void evolveSnakes(Point2i center, int radius, vector<Point2i> snake){

    Mat imageFloat;
    image.convertTo(imageFloat, CV_32FC3, (1.f/255.f));

    int verticesNumber =  radius*CV_PI/7;
    snake.resize(verticesNumber);

    float angle = 0.f;

    for (cv::Point2i& s : snake) {

        s.x = round(center.x + cos(angle) * radius);
        s.y = round(center.y + sin(angle) * radius);
        angle += 2*CV_PI/verticesNumber;
    }

    Mat vis;
    imageFloat.copyTo( vis );
    drawSnake(vis, snake);
    imshow("Snake", vis);
    waitKey(0);

    // Gradient images
    cv::Mat gx, gy, img_gray;
    cvtColor(imageFloat, img_gray, CV_BGR2GRAY);
    Sobel(img_gray, gx, -1, 1, 0, 3, 1./8.);
    Sobel(img_gray, gy, -1, 0, 1, 3, 1./8.);

    //blur the edges
    GaussianBlur(gx, gx, cv::Size(), 5);
    GaussianBlur(gy, gy, cv::Size(), 5);

  }


  void drawSnake(Mat img, vector<cv::Point2i>& snake){

    int siz = snake.size();

    for (size_t iii=0; iii<siz; iii++){
        
        line(img, snake[iii], snake[(iii+1)%siz], Scalar(0,0,1) );
      }

    for (const cv::Point2i& p: snake){

        circle(img, p, 2, cv::Scalar(1,0,0), -1 );      
    }

    }

  ~Snakes(){

    //nothing to do here

  }

};
