#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

const char* keys_surf =
        "{ help h |                          | Print help message. }"
        "{ input1 | ./me.bmp          | Path to input image 1. }"
        "{ input2 | ./me_in_scene3.bmp | Path to input image 2. }";

int main_surf( int argc, char* argv[] )
{
    CommandLineParser parser( argc, argv, keys_surf );
    Mat img_object = imread( parser.get<String>("input1"), IMREAD_GRAYSCALE );
    Mat img_scene = imread( parser.get<String>("input2"), IMREAD_GRAYSCALE );
    if ( img_object.empty() || img_scene.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }
	UMat uo, us;
	img_object.copyTo(uo);
	img_scene.copyTo(us);

	Mat obj_canny, scene_canny;


	cv::Canny(img_object, obj_canny, 147, 220, 3, true); // 1.5:1 or 3:2
	cv::Canny(img_scene, scene_canny, 147, 220, 3, true); // 1.5:1 or 3:2

	double dFrq = cv::getTickFrequency();
	int64 i64Begin = cv::getTickCount();

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );
	//detector->detectAndCompute( uo, noArray(), keypoints_object, descriptors_object );
	//detector->detectAndCompute( us, noArray(), keypoints_scene, descriptors_scene );
	//detector->detectAndCompute( obj_canny, noArray(), keypoints_object, descriptors_object ); // 以canny 图像为输入会更加耗时1.7s --> 2.9s
	//detector->detectAndCompute( scene_canny, noArray(), keypoints_scene, descriptors_scene );

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
	int64 i64Cur = cv::getTickCount();
	int64 i64TickCount = i64Cur - i64Begin;
	std::cout << "tick count: " << i64TickCount << std::endl;
	std::cout << "tick frequency: " << dFrq << std::endl;
	std::cout << "time(s): " << i64TickCount / dFrq << std::endl;
    //-- Draw matches
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    Mat H = findHomography( obj, scene, RANSAC );

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)img_object.cols, 0 );
    obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
    obj_corners[3] = Point2f( 0, (float)img_object.rows );
    std::vector<Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
          scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
          scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
          scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
          scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );

    //-- Show detected matches
    imshow("Good Matches & Object detection", img_matches );

	//imshow("obj_canny", obj_canny);
	//imshow("scene_canny", scene_canny);

    waitKey();
    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif
