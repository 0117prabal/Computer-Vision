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

    int kkkTOTAL = 9;

    // iterate until
    // - optimal solution for every point is the center of the box, OR
    // - until maximum number of iterations is reached
    int iii = 0, nodesChanged;

    do{
        iii++;
        nodesChanged = 0;

        const float dist = averageDistance(snake);
        Mat_ <float> energies = Mat::zeros(kkkTOTAL, verticesNumber, CV_32FC1); // this will hold the costs for each node and each state
        Mat_ <int>   position = Mat::zeros(kkkTOTAL, verticesNumber, CV_32SC1); // this will hold information about the minimum cost route to reach each node

        for (int vvv=0; vvv < verticesNumber; vvv++){
          // test all possible states (local neighborhood) of current (ccc) node
          for (int ccc=0; ccc<kkkTOTAL; ccc++){

            int cccOFF_Y, cccOFF_X;
            define_OFFs(ccc, cccOFF_Y, cccOFF_X);

            float termUNARY = term_Unary(Point2i( snake[vvv].x+cccOFF_X, snake[vvv].y+cccOFF_Y ), gx, gy);

            energies(ccc,vvv) = FLT_MAX;
            position(ccc,vvv) = -1;
            // test all possible states (local neighborhood) of previous (ppp) node
            for (int ppp=0; ppp < kkkTOTAL; ppp++){

              int pppOFF_Y, pppOFF_X;
              define_OFFs(ppp, pppOFF_Y, pppOFF_X);
              // test all possible states (local neighborhood) of next (nnn) node
              for (int nnn=0; nnn<kkkTOTAL; nnn++){

                  int nnnOFF_Y, nnnOFF_X;
                  define_OFFs(nnn, nnnOFF_Y, nnnOFF_X);
                  int VVV = verticesNumber;

                  double energyTMP; 

                  if(vvv==0){

                    energyTMP = termUNARY + energies(ppp,vvv-1) + term_Binary(Point2i(snake[vvv+0].x+cccOFF_X, snake[vvv+0].y+cccOFF_Y ), Point2i(snake[VVV-1].x+pppOFF_X, snake[VVV-1].y+pppOFF_Y), Point2i(snake[vvv+1].x+nnnOFF_X, snake[vvv+1].y+nnnOFF_Y), 0.00001, 0.000005, dist);

                  }               
                  
                  else if(vvv==VVV-1){

                    energyTMP = termUNARY + energies(ppp,vvv-1) + term_Binary(Point2i(snake[vvv+0].x+cccOFF_X, snake[vvv+0].y+cccOFF_Y), Point2i(snake[vvv-1].x+pppOFF_X, snake[vvv-1].y+pppOFF_Y), Point2i(snake[0].x+nnnOFF_X, snake[0].y+nnnOFF_Y ), 0.00001, 0.000005, dist);

                  }

                  else{

                    energyTMP = termUNARY + energies(ppp,vvv-1) + term_Binary(  cv::Point2i( snake[vvv+0].x+cccOFF_X, snake[vvv+0].y+cccOFF_Y ), Point2i( snake[vvv-1].x+pppOFF_X, snake[vvv-1].y+pppOFF_Y ), Point2i( snake[vvv+1].x+nnnOFF_X, snake[vvv+1].y+nnnOFF_Y ), 0.00001, 0.000005, dist);
                    
                  }

                  
                  if (energies(ccc,vvv) > energyTMP){

                   energies(ccc,vvv) = energyTMP;
                   position(ccc,vvv) = ppp;

                  }

              }
            }
          }
        }


        int posID = find_MIN_ID_inColumn(energies, energies.cols-1);

        for (int vvv=verticesNumber-1; vvv>=0; vvv--){

            posID = position(posID,vvv);
            int off_Y, off_X;
            define_OFFs(posID, off_Y, off_X);
            
            int prevID;

            if (vvv>0)
              prevID = vvv-1;
            else
              prevID = verticesNumber-1;
            
            snake[prevID].x += off_X;
            snake[prevID].y += off_Y;

            if (off_X != 0 || off_Y != 0)
              nodesChanged++;
        }

        cout << iii << ":\t" << nodesChanged << " nodes of the snake changed" <<endl;
        imageFloat.copyTo(vis);
        drawSnake(vis, snake);
        imshow("Snake", vis);
        waitKey(10);


      } while(iii < 10000 && nodesChanged > 0);

       waitKey();
       destroyAllWindows();

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


  float averageDistance(vector<Point2i> & snake){
    
    const int siz = snake.size();
    float dist = 0;

    for(int i = 0; i < snake.size(); i++){
           
           dist += sqrt(pow(snake[(i+1)%siz].x - snake[i].x, 2) + pow(snake[(i+1)%siz].y - snake[i].y, 2));

      }

    return dist / (float)siz;

  }

  void define_OFFs( const int& kkk, int& off_Y, int& off_X ){
    
    switch (kkk){ 
           //             i            j
        case 0:
            off_Y =  0;
            off_X =  0;
            break; ///
        
        case 1:
            off_Y =  0;
            off_X = -1;
            break;
        
        case 2:
            off_Y =  0;  
            off_X = +1;  
            break;

        case 3:
            off_Y = -1;
            off_X =  0;  
            break;

        case 4: 
            off_Y = -1;  
            off_X = -1;  
            break;

        case 5:
            off_Y = -1;  
            off_X = +1;  
            break;

        case 6: 
            off_Y = +1;  
            off_X =  0;  
            break;

        case 7:
            off_Y = +1;  
            off_X = -1;  
            break;

        case 8:
            off_Y = +1;  
            off_X = +1;  
            break;
    }

  }


  float term_Unary(const Point2i & p_curr, const Mat_<float> & gx, const Mat_<float> & gy){
    
    return -(pow(gx(p_curr), 2) + pow(gy(p_curr), 2));

  }

  float term_Binary(const Point2i& p_curr, const Point2i& p_prev, const Point2i& p_next, const float alpha, const float beta, const float dist){

       return term_Binary_Elasticity(p_curr, p_prev, alpha, dist) + term_Binary_Curvature(p_curr, p_prev, p_next, beta);
  }

  float term_Binary_Elasticity(const Point2i& p_curr, const Point2i& p_prev, const float alpha, const float dist){

       return alpha * pow(sqrt(pow(p_prev.x - p_curr.x, 2) + pow(p_prev.y - p_curr.y, 2) ) - dist, 2);
  }

  float term_Binary_Curvature(const Point2i& p_curr, const Point2i& p_prev, const Point2i& p_next, const float beta){

       return beta * (pow(p_next.x - 2 * p_curr.x + p_prev.x, 2) + pow(p_next.y - 2 * p_curr.y + p_prev.y, 2));
  }

  int find_MIN_ID_inColumn(const Mat_<float>& mat, int colID){

      float currMIN_Val = +999999;
      float currMIN_IDD = +999999;

      for (int iii=0; iii<mat.rows; iii++){

          if (currMIN_Val > mat(iii,colID)){
            
            currMIN_Val = mat(iii,colID);
            currMIN_IDD = iii;

          }
      }

      return currMIN_IDD;
  }

  ~Snakes(){

    //nothing to do here

  }

};
