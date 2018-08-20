// Created By: Rodrigo Gomez-Palacio, Juan Vasquez, Adolfo Portilla, Muhammad Ashfaq

#include <iostream>

// ROS includes
#include "ros/ros.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float32.h"

// Image includes
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <img_processing/segmentation.h>
#include <img_processing/colorConversion.h>
#include <img_processing/imageProcessing.h>
#include <img_processing/contour.h>

// Rodrigo includes
#include <sstream>
#include <iostream>
#include <fstream>
#include <sys/wait.h>
#include <thread>
#include <cmath>
#include <time.h>

// namespaces
using namespace std;
using namespace cv;

class SignDetectionROSNode{
public:
    // Constructor
    SignDetectionROSNode(int argc, char **argv);

private:
    // Setup variables
    bool DEBUG;
    int count;

    // Sign detection variables
    Mat input_image, final_image;
    Mat output_image, detected_edges;
    Mat red, yellow, light_yellow, white;
    vector<Mat> filter_vec;
    vector<Rect> bounding_boxes;
    char * window_name = (char*) "Filter";
    const int BOX_WIDTH = 3;                  /***--- Change Width Here--***/
    int filter_count;	                      //0 = red, 1 = yellow, 2 = white

    // ROS output variables
    std_msgs::Bool stopDetected;
    std_msgs::Float32 stopDistance;
    std_msgs::Float32 stopPercentage;

    // ROS publisher variables
    ros::Publisher stop_detected_pub;
    ros::Publisher stop_distance_pub;
    // ros::Publisher yield_detected_pub;
    // ros::Publisher yield_distance_pub;
    // ros::Publisher speed_detected_pub;
    // ros::Publisher speed_distance_pub;

    // Show OpenCV image
    void show_single_image(string, Mat &img);

    // Front camera callback function
    void frontCameraCallback(const sensor_msgs::ImageConstPtr& msg);

    // Sign detection pipeline
    void sign_detection_pipeline(Mat &input_image);

    // Neural network callback
    void neuralCallback();

    // Sign filter function
    Mat sign_filter(int, void*);

    // Is symmertric
    bool is_symmetric(Rect r);

    // Is large
    bool is_large(Rect r);

    // Is small
    bool is_small(Rect r);

    // Change hue
    void hue_shift(Mat &image);
};


// Constructor
SignDetectionROSNode::SignDetectionROSNode(int argc, char **argv){
    // initalize setup variables
    this->DEBUG = true;
    this->count = 0;
    this->stopDetected.data = false;
    this->stopDistance.data = 0.0;
    this->stopPercentage.data = 0.0;

    // initialize ROS node
    ros::init(argc, argv, "Sign_Detection_Subscriber");

    // initalize ROS node handler
    ros::NodeHandle n;

    // ROS setup
    this->stop_detected_pub = n.advertise<std_msgs::Bool>("/sign_detection/stop_sign_detected", 1000);
    this->stop_distance_pub = n.advertise<std_msgs::Float32>("/sign_detection/stop_sign_distance", 1000);

    // Setup image transport
    image_transport::ImageTransport it(n);

    // initialize ROS subscriber
    image_transport::Subscriber image_sub = it.subscribe("/front_camera/image_raw", 1, &SignDetectionROSNode::frontCameraCallback, this);

    // ensure subscribers do not stop
    ros::spin();
}

// Show OpenCV image
void SignDetectionROSNode::show_single_image(string name, Mat &img){
    namedWindow(name, WINDOW_NORMAL);
    resizeWindow(name, 312, 312);

    imshow(name, img);
    waitKey(1);
}

// Front camera callback function
void SignDetectionROSNode::frontCameraCallback(const sensor_msgs::ImageConstPtr& msg){
    // recieved image
    this->input_image = cv_bridge::toCvShare(msg, "bgr8")->image;

    // assert image
    CV_Assert(input_image.channels() == 3);

    // Check that the image has been opened
    if (!this->input_image.data) {
       cout << "Error to read the image. Check ''cv::imread'' function of OpenCV." << endl;
       return;
    }

    // resize
    resize(this->input_image, this->input_image, cvSize(1024, 1024));

    // show the input image
    if(this->DEBUG){
        //this->show_single_image("Initial Input Image", this->input_image);
    }

    // start sign_detection pipeline (assuming valid image)
    this->sign_detection_pipeline(this->input_image);
}

// Sign Detection pipeline
void SignDetectionROSNode::sign_detection_pipeline(Mat &img){
    // output to terminal
    this->count++;
    if(this->DEBUG)
        cout << "[" << this->count << "] "; // << "Starting Pipeline..." << endl;

    // 0. reset variables
    this->filter_vec = {};
    this->bounding_boxes = {};
    this->filter_count = 0;	    //0 = red, 1 = yellow, 2 = white
    this->input_image = img;
    //hue_shift(this->input_image);
    //this->input_image.convertTo(this->input_image, CV_8UC1,5, 30);
    this->final_image = img;
    this->stopDetected.data = false;
    this->stopDistance.data = -1.0;
    this->stopPercentage.data = 0.0;

    // filters for each color
    red = sign_filter(0, 0);
    //yellow = sign_filter(0, 0);
    //light_yellow = sign_filter(0, 0);
    //white = sign_filter(0, 0);

    filter_vec = {red}; // include yellow, etc. in future
    vector<vector<Point>> contours;	//final contours

    for(int i = 0; i < 1; i++){ // number of sign filters to detect
        Mat with_contours = this->input_image;
        //imshow("input", this->input_image);
        vector<vector<Point>> temp_contours;
        findContours(filter_vec[i], temp_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // approximate contours
        contours = {};
        for(unsigned int j = 0; j < temp_contours.size(); j++){
            vector<Point> poly;
            approxPolyDP(temp_contours[j], poly, 1, true);
            contours.push_back(poly);
        }

        // bounding Rectangles
        this->bounding_boxes = {};
        for(unsigned int j=0; j<contours.size(); j++){
            Rect box = boundingRect(contours[j]);
            if(is_symmetric(box) && !is_small(box) && !is_large(box)){
                this->bounding_boxes.push_back(box);
            }
        }
        
        // neural net
        if(this->DEBUG)
            cout << "\tNumber of Possible Stop Signs: " << this->bounding_boxes.size() << endl;
            
        for(unsigned int i = 0; i < bounding_boxes.size(); i++){
            //crop image and put in bucket
            Rect box = bounding_boxes[i];   //get bounding box
            
            // we will now resize the image
            Mat crop = final_image(box);

            int height = crop.size().height;
            int width = crop.size().width;

            // resize image to 32x32
            resize(crop, crop, cvSize(32, 32));

            // attempt for better images
            // histogram equalization
            Mat src = imread("/home/shared/TAMU_AutoDrive_Year_1/catkin_ws/src/stop_signs/bucket/base.jpg", 1);
            Mat hist_equalized_image;
            cvtColor(crop, hist_equalized_image , COLOR_BGR2YCrCb);
            vector<Mat> vec_channels;
            split(hist_equalized_image, vec_channels);
            equalizeHist(vec_channels[0],vec_channels[0]);
            merge(vec_channels,hist_equalized_image);
            cvtColor(hist_equalized_image,hist_equalized_image,COLOR_YCrCb2BGR);
 


            // save image
            string filepath = "/home/shared/TAMU_AutoDrive_Year_1/catkin_ws/src/stop_signs/bucket/image.jpg";
            //imwrite(filepath, crop);
            imwrite(filepath, hist_equalized_image);
            rectangle(with_contours, box, Scalar(0,255,0), this->BOX_WIDTH);

            // start nerual net thread
            thread neuralNetThread (&SignDetectionROSNode::neuralCallback, this);

            // wait for nerual net to finish
            neuralNetThread.join();

            //check txt file to see if true
            ifstream bucket;
            bucket.open("/home/shared/TAMU_AutoDrive_Year_1/catkin_ws/src/stop_signs/bucket/image.txt");

            // variables within bucket
	        string percentage;
            bool is_sign;
            
            // save bucket variables
            bucket >> boolalpha >> is_sign;
	        bucket >> percentage;
            bucket.close();

            //if true:
            if(is_sign){
                if(this->DEBUG){
                    cout << "\t\tSTOP SIGN DETECTED: " << percentage << endl;
                    cout << "Height: " << height << ", Width: " << width << endl;
                }
                
                // set ROS message values
                this->stopDetected.data = true;
                this->stopPercentage.data = strtof(percentage.c_str(), 0);
                this->stopDistance.data = (float) (1662.8 * pow(height * 2.0, -1.136));	

            } else{
                if(this->DEBUG){
                    cout<<"\t\tNOPE: " << percentage << endl;
                    cout << "Height: " << height << ", Width: " << width << endl;
                }

                if(this->stopDetected.data == 0.0){
                    this->stopPercentage.data = strtof(percentage.c_str(), 0);
                    this->stopDistance.data = -1.0;
                }
            }
        }
    }
    
    // publish to ROS topics (this considers any possible true values)
    this->stop_detected_pub.publish(this->stopDetected);
    this->stop_distance_pub.publish(this->stopDistance);
    if(this->DEBUG){
        this->show_single_image("Sign Detection Output", this->input_image);
        cout << "\tFinal Judgement: " << (bool) this->stopDetected.data << " with certainty " << this->stopPercentage.data << endl;
    }
}

void SignDetectionROSNode::neuralCallback(){
    //cout << "\tCalling Neural Network... ";		
    system("mogrify -format bmp /home/shared/TAMU_AutoDrive_Year_1/catkin_ws/src/stop_signs/bucket/image.jpg");                 
    system("/home/shared/TAMU_AutoDrive_Year_1/catkin_ws/src/stop_signs/neural2d/build/neural2d > /dev/null");
}

bool SignDetectionROSNode::is_large(Rect r){
    if(r.height > this->input_image.size().height*.2 || r.width > this->input_image.size().width*.2){
        return 1;
    } else return 0;
}

bool SignDetectionROSNode::is_small(Rect r){
    if(r.height < this->input_image.size().height*.005 || r.width < this->input_image.size().width*.005){
        return 1;
    } else return 0;
}

bool SignDetectionROSNode::is_symmetric(Rect r){
    //Discards many boxes because they can't be signs
    if(r.height*1.5 < r.width) return 0;
    else if(r.width*1.5 < r.height) return 0;
    else return 1;
}

Mat SignDetectionROSNode::sign_filter(int, void*){
    Mat hsv_image;
    cvtColor(this->input_image, hsv_image, COLOR_BGR2HSV);

    if(this->filter_count == 0){
        //keep red pixels
        Mat lower_red;
        Mat upper_red;

        inRange(hsv_image, Scalar(0,30,30), Scalar(10,255,255), lower_red);     // Juan - changed from 80 to 70
        inRange(hsv_image, Scalar(160,70,30), Scalar(179, 255,255), upper_red);

        Mat red_image;
        addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0, red_image);


        imageprocessing::filter_image(red_image, this->output_image);
    } 

    Mat *temp = new Mat(this->output_image);
    return *temp;
}

void SignDetectionROSNode::hue_shift(Mat &image){
	Mat hsv;
	cvtColor(image, hsv, CV_RGB2HSV);
	for(int c=0; c<hsv.cols; c++){
		for(int r=0; r<hsv.rows; r++){
			Vec3b &point = hsv.at<Vec3b>(Point(c,r));
			int value = 0;				
			if(point[0] < 130){
				int hue = hsv.at<Vec3b>(Point(c,r))[0];
				int saturation = hsv.at<Vec3b>(Point(c,r))[1];
				
				if(point[0] > 100){
					hue = 120;						
					value = 255;
					saturation = 200;
				

				}

				hsv.at<Vec3b>(Point(c,r))[0] = hue;
				hsv.at<Vec3b>(Point(c,r))[1] = saturation;
			}

		}		
	}

	cvtColor(hsv, image, CV_HSV2RGB);
}

// Main function
int main(int argc, char **argv){    
    // welcome
    cout << "Starting Sign Detection ROS Node Class..." << endl;

    // instantiate ROS node for Sign Detection
    SignDetectionROSNode ROSNode(argc, argv);

    // output to terminal
    cout << "Sign Detection ROS Node Exiting..." << endl;

    // return successfully
    return 0;
}
