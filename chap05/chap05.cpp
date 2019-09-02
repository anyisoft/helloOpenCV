// chap05.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
/*
// alpha blend
int main(int argc, char **argv)
{
	cv::Mat src1 = cv::imread(argv[1], 1);
	cv::Mat src2 = cv::imread(argv[2], 1);

	if (argc != 9 || src1.empty() || src2.empty()) {
		std::cout << "Invalid params!" << std::endl;
		return 1;
	}

	int x = atoi(argv[3]);
	int y = atoi(argv[4]);
	int w = atoi(argv[5]);
	int h = atoi(argv[6]);
	double alpha = atoi(argv[7]);
	double beta = atoi(argv[8]);

	cv::Mat roi1(src1, cv::Rect(x, y, w, h));
	cv::Mat roi2(src2, cv::Rect(550, 400, w, h));

	cv::addWeighted(roi1, alpha, roi2, beta, 0.0, roi1);

	cv::namedWindow("Alpha Blend", 1);
	cv::imshow("Alpha Blend", src1);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 5-1
int main()
{
	cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);
	cv::Point center(50, 50);
	cv::Scalar color(255, 0, 0, 0);
	cv::circle(img, center, 20, color);

	cv::namedWindow("Exercise5-1", 1);
	cv::imshow("Exercise5-1", img);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 5-3
int main()
{
	cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);
	
	for (int i = 20; i <= 40; i++) {
		for (int j = 5; j <= 20; j++) {
			img.at<cv::Vec3b>(j, i)[1] = 255;
		}
	}

	cv::namedWindow("Exercise5-2", 1);
	cv::imshow("Exercise5-2", img);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 5-3
int main()
{
	cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);

	for (int i = 5; i <= 20; i++) {
		for (int j = 20; j <= 40; j++) {
			*((uchar*)(img.ptr<cv::Vec3b>(i, j))+1) = 255;
		}
	}
	
	cv::namedWindow("Exercise5-3", 1);
	cv::imshow("Exercise5-3", img);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 5-4
int main()
{
	cv::Mat img = cv::Mat::zeros(210, 210, CV_8UC1);
	cv::Mat mask = cv::Mat::zeros(210, 210, CV_8U);
	
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j <= (i+1)*20; j++) {
			for (int k = 0; k < 10; k++) {
				*mask.ptr<cv::Vec3b>(i*20+k, j) = 1;
			}
		}
	}
	
	img.setTo(192, mask);

	
	cv::namedWindow("Exercise5-3", 1);
	cv::imshow("Exercise5-3", img);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise5-5
int main(int argc, char** argv)
{
	cv::Mat src = cv::imread(argv[1], 1);

	cv::Mat roi1(src, cv::Rect(5, 10, 20, 30));
	cv::Mat roi2(src, cv::Rect(50, 60, 20, 30));

	cv::bitwise_not(roi1, roi1);
	cv::bitwise_not(roi2, roi2);

	cv::namedWindow("Exercise5-5", 1);
	cv::imshow("Exercise5-5", src);

	cv::waitKey(0);

	return 0;
}
*/
// Exercise5-6
// 本题目解答参考了博客
// https://blog.csdn.net/guduruyu/article/details/70837779
//
int main(int argc, char** argv)
{
	cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
	
	std::vector<cv::Mat> rgbChannels;
	cv::split(src, rgbChannels);

	cv::Mat blank_ch, fin_img;
	blank_ch = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

	std::vector<cv::Mat> channels_r;
	channels_r.push_back(blank_ch);
	channels_r.push_back(blank_ch);
	channels_r.push_back(rgbChannels[2]);
	cv::merge(channels_r, fin_img);
	cv::namedWindow("Exercise5-6-r", 1);
	cv::imshow("Exercise5-6-r", fin_img);

	std::vector<cv::Mat> channels_g;
	channels_g.push_back(blank_ch);
	channels_g.push_back(rgbChannels[1]);
	channels_g.push_back(blank_ch);
	cv::merge(channels_g, fin_img);
	cv::namedWindow("Exercise5-6-g", 1);
	cv::imshow("Exercise5-6-g", fin_img);

	cv::Mat clone1 = fin_img.clone();
	cv::Mat clone2 = fin_img.clone();

	double minVal = 0;
	double maxVal = 0;
	int minIdx = 0;
	int maxIdx = 0;
	cv::minMaxIdx(rgbChannels[1], &minVal, &maxVal, &minIdx, &maxIdx);
	std::cout << "minVal: " << minVal << " maxVal: " << maxVal
		<< " minIdx: " << minIdx << " maxIdx: " << maxIdx << std::endl;

	cv::Point minLoc;
	cv::Point maxLoc;
	cv::minMaxLoc(rgbChannels[1], &minVal, &maxVal, &minLoc, &maxLoc);
	std::cout << "minVal: " << minVal << " maxVal: " << maxVal
		<< " minLoc: " << minLoc << " maxLoc: " << maxLoc << std::endl;

	uchar thresh = (uchar)((maxVal - minVal) / 2.0);
	clone1.setTo(thresh);

	clone2.setTo(0);
	cv::compare(fin_img, clone1, clone2, cv::CMP_GE);

	//cv::subtract(fin_img, clone1 / 2, fin_img, clone2);
	//cv::subtract(fin_img, clone1 / 2, fin_img);
	cv::subtract(fin_img, thresh / 2, fin_img);
	cv::namedWindow("Exercise5-6-f", 1);
	cv::imshow("Exercise5-6-f", fin_img);

	cv::namedWindow("clone1", 1);
	cv::imshow("clone1", clone1);

	cv::namedWindow("clone2", 1);
	cv::imshow("clone2", clone2);

	std::vector<cv::Mat> channels_b;
	channels_b.push_back(rgbChannels[0]);
	channels_b.push_back(blank_ch);
	channels_b.push_back(blank_ch);
	cv::merge(channels_b, fin_img);
	cv::namedWindow("Exercise5-6-b", 1);
	cv::imshow("Exercise5-6-b", fin_img);

	cv::waitKey(0);

	return 0;
}
// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
