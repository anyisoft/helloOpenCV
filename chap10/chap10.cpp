// chap10.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
/*
// Example 10-1
// Example 10-2
void sum_rgb(const cv::Mat& src, cv::Mat& dst)
{
	std::vector<cv::Mat> planes;
	cv::split(src, planes);

	cv::Mat b = planes[0];
	cv::Mat g = planes[1];
	cv::Mat r = planes[2];
	cv::Mat s;

	cv::addWeighted(r, 1. / 3., g, 1. / 3., 0.0, s);
	cv::addWeighted(s, 1., b, 1. / 3., 0.0, s);

	cv::threshold(s, dst, 100, 100, cv::THRESH_TRUNC);
}

void sum_rgbV2(const cv::Mat& src, cv::Mat& dst)
{
	std::vector<cv::Mat> planes;
	cv::split(src, planes);

	cv::Mat b = planes[0];
	cv::Mat g = planes[1];
	cv::Mat r = planes[2];
	cv::Mat s = cv::Mat::zeros(b.size(), CV_32F);

	cv::accumulate(b, s);
	cv::accumulate(g, s);
	cv::accumulate(r, s);

	//cv::threshold(s, s, 100, 100, cv::THRESH_TRUNC);
	cv::threshold(s, s, 100, 100, cv::THRESH_OTSU);
	s.convertTo(dst, b.type());
}

void help()
{
	std::cout << "Call: ./ch10_ex10_1 faceScene.jpg" << std::endl;
	std::cout << "Shows use of alpha blending (addWeighted) and threshold" 
		<< std::endl;
}

int main(int argc, char **argv)
{
	help();

	if (argc < 2) {
		std::cout << "specify input image" << std::endl;
		return -1;
	}

	//cv::Mat src = cv::imread(argv[1]);
	cv::Mat src = cv::imread("1.bmp");
	cv::Mat dst;

	if (src.empty()) {
		std::cout << "can not load " << argv[1] << std::endl;
		return -1;
	}

	sum_rgb(src, dst);
	//sum_rgbV2(src, dst);

	//cv::imshow(argv[1], dst);
	cv::imshow("1", dst);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Example 10-3
int main(int argc, char** argv)
{
	if (argc != 7) {
		std::cout << 
			"Usage: " << argv[0] << " fixed_threshold invert(0=off|1=on)"
			"adaptive_type(0=mean|1=gaussian) block_size offset image\n"
			"Example: " << argv[0] << " 100 1 0 15 10 fruits.jpg\n";
		return -1;
	}

	double fixed_threshold = (double)atof(argv[1]);
	int threshold_type = atoi(argv[2]) ? 
		cv::THRESH_BINARY : cv::THRESH_BINARY_INV;
	int adaptive_method = atoi(argv[3]) ? 
		cv::ADAPTIVE_THRESH_MEAN_C : cv::ADAPTIVE_THRESH_GAUSSIAN_C;
	int block_size = atoi(argv[4]);
	double offset = (double)atof(argv[5]);

	cv::Mat Igray = cv::imread(argv[6], cv::IMREAD_GRAYSCALE);

	if (Igray.empty()) {
		std::cout << "Can not load " << argv[6] << std::endl;
		return -1;
	}

	cv::Mat It, Iat;

	cv::threshold(Igray, It, fixed_threshold, 255, threshold_type);

	cv::adaptiveThreshold(Igray, Iat, 255, adaptive_method, threshold_type, block_size, offset);

	cv::imshow("Raw", Igray);
	cv::imshow("Threshold", It);
	cv::imshow("Adaptive Threshold", Iat);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-1
// a.由3*3 --> 5*5 --> 9*9 --> 11*11，越来越模糊
// b.不相同，因为5*5 核更小，即使平滑两次，也比一次9*9 的清楚，
//   更不要说11*11。进一步发现，同一核即使重复平滑多次也没什么变化
int main()
{
	cv::Mat raw, gb33, gb55, gb55_2, gb99, gb1111;

	//raw = cv::imread("10-1.bmp");
	raw = cv::imread("1.bmp");


	cv::GaussianBlur(raw, gb33, cv::Size(3, 3), 0);
	cv::GaussianBlur(raw, gb55, cv::Size(5, 5), 0);
	cv::GaussianBlur(gb55, gb55_2, cv::Size(5, 5), 0);
	for (int i = 0; i < 10; i++) {
		cv::GaussianBlur(gb55_2, gb55_2, cv::Size(5, 5), 0);
	}
	cv::GaussianBlur(raw, gb99, cv::Size(9, 9), 0);
	cv::GaussianBlur(raw, gb1111, cv::Size(11, 11), 0);

	cv::imshow("gb33", gb33);
	cv::imshow("gb55", gb55);
	cv::imshow("gb55_2", gb55);
	cv::imshow("gb99", gb99);
	cv::imshow("gb1111", gb1111);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-2
// a b
// 平滑后肉眼基本观测不到可见内容
// 直接察看平滑结果的图片数据，发现在图片中间区域有一小块非0 数据
// 5*5 时，sum = 256
// 1,  4,  6,  4, 1
// 4, 16, 24, 16, 4
// 6, 24, 36, 24, 6
// 4, 16, 24, 16, 4
// 1,  4,  6,  4, 1
// 5*5 两次，sum = 251
// 0, 1,  2,  2, 2,  1, 0
// 1, 3,  6,  8, 6,  3, 1
// 2, 6, 12, 15, 12, 6, 2
// 2, 8, 15, 19, 15, 8, 2
// 2, 6, 12, 15, 12, 6, 2
// 1, 3,  6,  8,  6, 3, 1
// 0, 1,  2,  2,  2, 1, 0
// 9*9 时，sum=262
// 0, 0, 0,  1,  1,  1, 0, 0, 0
// 0, 1, 2,  3,  3,  3, 2, 1, 0
// 0, 2, 4,  6,  7,  6, 4, 2, 0
// 1, 3, 6, 10, 12, 10, 6, 3, 1
// 1, 3, 7, 12, 14, 12, 7, 3, 1
// 1, 3, 6, 10, 12, 10, 6, 3, 1
// 0, 2, 4,  6,  7,  6, 4, 2, 0
// 0, 1, 2,  3,  3,  3, 2, 1, 0
// 0, 0, 0,  1,  1,  1, 0, 0, 0
// 9*9 两次, sum=255
// 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0
// 0, 0, 1, 1, 2, 2, 2, 1, 1, 0, 0
// 0, 1, 2, 2, 3, 3, 3, 2, 2, 1, 0
// 1, 1, 2, 4, 5, 5, 5, 4, 2, 1, 1
// 1, 2, 3, 5, 6, 7, 6, 5, 3, 2, 1
// 1, 2, 3, 5, 7, 7, 7, 5, 3, 2, 1
// 1, 2, 3, 5, 6, 7, 6, 5, 3, 2, 1
// 1, 1, 2, 4, 5, 5, 5, 4, 2, 1, 1
// 0, 1, 2, 2, 3, 3, 3, 2, 2, 1, 0
// 0, 0, 1, 1, 2, 2, 2, 1, 1, 0, 0
// 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0
// 5*5 两次当然与9*9 不同，影响范围就不一样
int main()
{
	//int64 gb55DataAddr = 0;
	cv::Mat raw, gb55, gb55_2, gb99, gb99_2;

	raw = cv::Mat::zeros(100, 100, CV_8UC1);

	raw.at<uchar>(50, 50) = 255;

	cv::GaussianBlur(raw, gb55, cv::Size(5, 5), 0);
	//gb55DataAddr = (int64)gb55.data;
	//std::cout << "gb55_data: " << std::hex << gb55DataAddr << " gb55:" << gb55 << std::endl;
	std::cout << "gb_55 sum: " << cv::sum(gb55) << " gb55_kernel\n" << gb55.rowRange(45, 55).colRange(45, 55) << std::endl;
	cv::GaussianBlur(gb55, gb55_2, cv::Size(5, 5), 0);
	std::cout << "gb_55_2 sum: " << cv::sum(gb55_2) << " gb55_2_kernel\n" << gb55_2.rowRange(45, 55).colRange(45, 55) << std::endl;
	cv::GaussianBlur(raw, gb99, cv::Size(9, 9), 0);
	std::cout << "gb_99 sum: " << cv::sum(gb99) << " gb99_kernel\n" << gb99.rowRange(45, 55).colRange(45, 55) << std::endl;
	cv::GaussianBlur(gb99, gb99_2, cv::Size(9, 9), 0);
	std::cout << "gb_99_2 sum: " << cv::sum(gb99_2) << " gb99_2_kernel\n" << gb99_2.rowRange(40, 60).colRange(40, 60) << std::endl;
	
	cv::imshow("raw", raw);
	cv::imshow("gb55", gb55);
	cv::imshow("gb55_2", gb55_2);
	cv::imshow("gb99", gb99);
	cv::imshow("gb99_2", gb99_2);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-3
// 这里的param1、2、3，来源于OpenCV早期版本的cvSmooth 函数原型
// 对于GaussianBlur 函数而言，
// param1 = ksize.width, param2 = ksize.height
// param3 = sigmaX, param4 = sigmaY
int main()
{
	cv::Mat raw;
	cv::Mat gb33, gb33_s1, gb33_s4, gb33_s6;
	cv::Mat gb55, gb55_s1, gb55_s4, gb55_s6;
	cv::Mat gb99, gb99_s1, gb99_s4, gb99_s6;
	
	cv::Size size33(3, 3);
	cv::Size size55(5, 5);
	cv::Size size99(9, 9);
	cv::Size size00(0, 0);
	cv::Size curSize;

	//raw = cv::imread("10-1.bmp");
	raw = cv::imread("1.bmp");

	//curSize = size99;
	curSize = size00;
	//cv::GaussianBlur(raw, gb99, size99, 0);
	//cv::GaussianBlur(raw, gb99_s1, size99, 1);
	//cv::GaussianBlur(raw, gb99_s4, size99, 4);
	//cv::GaussianBlur(raw, gb99_s6, size99, 6);
	
	//cv::imshow("gb99", gb99);
	//cv::imshow("gb99_s1", gb99_s1);
	//cv::imshow("gb99_s4", gb99_s4);
	//cv::imshow("gb99_s6", gb99_s6);

	//cv::GaussianBlur(raw, gb55, size55, 0);
	//cv::GaussianBlur(raw, gb55_s1, size55, 1);
	//cv::GaussianBlur(raw, gb55_s4, size55, 4);
	//cv::GaussianBlur(raw, gb55_s6, size55, 6);

	//cv::imshow("gb55", gb55);
	//cv::imshow("gb55_s1", gb55_s1);
	//cv::imshow("gb55_s4", gb55_s4);
	//cv::imshow("gb55_s6", gb55_s6);

	cv::GaussianBlur(raw, gb33, size33, 0);
	cv::GaussianBlur(raw, gb33_s1, size33, 1);
	cv::GaussianBlur(raw, gb33_s4, size33, 4);
	cv::GaussianBlur(raw, gb33_s6, size33, 6);

	cv::imshow("gb33", gb33);
	cv::imshow("gb33_s1", gb33_s1);
	cv::imshow("gb33_s4", gb33_s4);
	cv::imshow("gb33_s6", gb33_s6);
	
	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-4
int main()
{
	cv::Mat src1, src2, diff12, cleandiff, dirtydiff, kernel, temp;
	src1 = cv::imread("10-4-1.jpg");
	src2 = cv::imread("10-4-2.jpg");

	diff12 = src1 - src2;

	cv::erode(diff12, temp, kernel);
	cv::dilate(temp, cleandiff, kernel);

	cv::dilate(diff12, temp, kernel);
	cv::erode(temp, dirtydiff, kernel);

	cv::imshow("diff12", diff12);
	cv::imshow("cleandiff", cleandiff);
	cv::imshow("dirtydiff", dirtydiff);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-5
int main()
{
	cv::Mat src1, src2, gray1, gray2;
	cv::Mat diff12, threshold50, opening, eroded, outline;
	cv::Mat element, temp;
	src1 = cv::imread("10-5-1.jpg");
	src2 = cv::imread("10-5-2.jpg");

	cv::cvtColor(src1, gray1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(src2, gray2, cv::COLOR_BGR2GRAY);

	diff12 = cv::abs(gray1 - gray2);

	cv::threshold(diff12, threshold50, 50., 255., cv::THRESH_BINARY);

	cv::morphologyEx(threshold50, opening, cv::MORPH_OPEN, element);

	cv::erode(opening, eroded, element);
	//cv::cvtColor(eroded, temp, cv::COLOR_GRAY2BGR);
	//cv::bitwise_xor(temp, src2, outline);
	cv::bitwise_xor(gray2, eroded, outline);

	cv::imshow("diff12", diff12);
	cv::imshow("threshold50", threshold50);
	cv::imshow("opening", opening);
	cv::imshow("eroded", eroded);
	cv::imshow("outline", outline);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-6
// 手写
int main()
{
	cv::Mat raw, balance;
	uchar r, g, b;
	uchar r_want, g_want, b_want;
	int thresH = 255*3;
	int thresL = 0;
	double rateH = 0.7;
	double rateL = 0.3;

	int max = 0, min = 0, temp = 0;

	raw = cv::imread("10-6-3.jpg");

	if (raw.empty()) {
		return 0;
	}

	raw.copyTo(balance);

	max = thresH * rateH;
	min = thresH * rateL;
	for (int i = 0; i < balance.rows; i++) {
		for (int j = 0; j < balance.cols; j++) {
			b = balance.at<cv::Vec3b>(i, j)[0];
			g = balance.at<cv::Vec3b>(i, j)[1];
			r = balance.at<cv::Vec3b>(i, j)[2];
			temp = r + g + b;
			if (temp > max) {
				b_want = b * (1 - (b * 1.0) / temp * (1 - rateH));
				g_want = g * (1 - (g * 1.0) / temp * (1 - rateH));
				r_want = r * (1 - (r * 1.0) / temp * (1 - rateH));

				balance.at<cv::Vec3b>(i, j)[0] = b_want;
				balance.at<cv::Vec3b>(i, j)[1] = g_want;
				balance.at<cv::Vec3b>(i, j)[2] = r_want;
			}
			else if (temp < min) {
				b_want = b * (1 + (b * 1.0) / temp * rateL);
				g_want = g * (1 + (g * 1.0) / temp * rateL);
				r_want = r * (1 + (r * 1.0) / temp * rateL);

				balance.at<cv::Vec3b>(i, j)[0] = b_want;
				balance.at<cv::Vec3b>(i, j)[1] = g_want;
				balance.at<cv::Vec3b>(i, j)[2] = r_want;
			}
		}
	}

	cv::imshow("raw", raw);
	cv::imshow("balance", balance);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-6
// 参考：https://blog.csdn.net/marooon/article/details/81560083
// 有bug?
void highlightRemove(cv::Mat& src, cv::Mat& dst)
{
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			double B = src.at<cv::Vec3b>(i, j)[0];
			double G = src.at<cv::Vec3b>(i, j)[1];
			double R = src.at<cv::Vec3b>(i, j)[2];

			double alpha_r = R / (R + G + B);
			double alpha_g = G / (R + G + B);
			double alpha_b = B / (R + G + B);

			double alpha =
				cv::max<double>(cv::max<double>(alpha_r, alpha_g), alpha_b);
			double maxC = cv::max<double>(cv::max<double>(R, G), B);
			double alpha_min =
				cv::min<double>(cv::min<double>(alpha_r, alpha_g), alpha_b);

			double beta_r = 1 - (alpha - alpha_r) / (3 * alpha - 1);
			double beta_g = 1 - (alpha - alpha_g) / (3 * alpha - 1);
			double beta_b = 1 - (alpha - alpha_b) / (3 * alpha - 1);
			double beta =
				cv::max<double>(cv::max<double>(beta_r, beta_g), beta_b);

			double gama_r = (alpha_r - alpha_min) / (1 - 3 * alpha_min);
			double gama_g = (alpha_g - alpha_min) / (1 - 3 * alpha_min);
			double gama_b = (alpha_b - alpha_min) / (1 - 3 * alpha_min);
			double gama =
				cv::max<double>(cv::max<double>(gama_r, gama_g), gama_b);

			double temp = (gama * (R + G + B) - maxC) / (3 * gama - 1);
			if (i < 10) {
				std::cout << " temp: " << temp;// << std::endl;
			}
			dst.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(B - (temp + 0.5));
			dst.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(G - (temp + 0.5));
			dst.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(R - (temp + 0.5));
		}
	}
}

int main()
{
	cv::Mat raw, balance;
	
	raw = cv::imread("10-6-4.jpg");
	
	if (raw.empty()) {
		return 0;
	}

	raw.copyTo(balance);

	highlightRemove(raw, balance);

	cv::imshow("raw", raw);
	cv::imshow("balance", balance);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-6
// threshold
int main()
{
	cv::Mat raw, balance;
	uchar r, g, b;
	uchar r_want, g_want, b_want;
	int thresH = 255 * 3;
	int thresL = 0;
	double rateH = 0.7;
	double rateL = 0.3;

	int max = 0, min = 0, temp = 0;

	raw = cv::imread("10-6-4.jpg");

	if (raw.empty()) {
		return 0;
	}

	raw.copyTo(balance);

	cv::threshold(raw, balance, 200, 220, cv::THRESH_OTSU);

	cv::imshow("raw", raw);
	cv::imshow("balance", balance);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-6
// 参考：https://www.cnblogs.com/ggYYa/p/5707259.html
//该代码实现白平衡算法中的灰度世界法，能有效改善图像发红发蓝发绿的现象；
//lw: 比较可以发现，此代码在图像有偏色时效果很好，对于光照不均匀并无帮助
using namespace std;
using namespace cv;

int main()
{
	Mat g_srcImage, dstImage;
	vector<Mat> g_vChannels;
	//g_srcImage = imread("10-6-1.jpg");
	//g_srcImage = imread("10-6-2.jpg");
	//g_srcImage = imread("10-6-3.jpg");
	//g_srcImage = imread("10-6-4.jpg");
	g_srcImage = imread("10-6-5.png");
	imshow("raw", g_srcImage);
	//waitKey(0);
	
	//分离通道
	split(g_srcImage, g_vChannels);
	Mat imageBlueChannel = g_vChannels.at(0);
	Mat imageGreenChannel = g_vChannels.at(1);
	Mat imageRedChannel = g_vChannels.at(2);

	double imageBlueChannelAvg = 0;
	double imageGreenChannelAvg = 0;
	double imageRedChannelAvg = 0;

	//求各通道的平均值
	imageBlueChannelAvg = mean(imageBlueChannel)[0];
	imageGreenChannelAvg = mean(imageGreenChannel)[0];
	imageRedChannelAvg = mean(imageRedChannel)[0];

	//求出个通道所占增益
	double K = (imageRedChannelAvg + imageGreenChannelAvg + imageRedChannelAvg) / 3;
	double Kb = K / imageBlueChannelAvg;
	double Kg = K / imageGreenChannelAvg;
	double Kr = K / imageRedChannelAvg;

	//更新白平衡后的各通道BGR值
	addWeighted(imageBlueChannel, Kb, 0, 0, 0, imageBlueChannel);
	addWeighted(imageGreenChannel, Kg, 0, 0, 0, imageGreenChannel);
	addWeighted(imageRedChannel, Kr, 0, 0, 0, imageRedChannel);

	merge(g_vChannels, dstImage);//图像各通道合并
	imshow("balance", dstImage);
	waitKey(0);
	return 0;
}
*/
/*
//自定义核
int main()
{
	cv::Mat src, dst;
	cv::Mat kernel;

	cv::Point anchor;

	double delta;
	int ddepth;
	int kernel_size;

	const char* window_name = "filter2d Demo";

	int c;

	src = cv::imread("10-6-4.jpg");
	if (src.empty()) {
		return -1;
	}

	anchor = cv::Point(-1, -1);
	delta = 0;
	ddepth = -1;

	int ind = 0;
	while (true) {
		// 每隔0.5s，用一个不同的核来对图像进行滤波
		c = cv::waitKey(5000);
		if (27 == (char)c) {
			break;
		}

		// 更新归一化块滤波器的核大小
		kernel_size = 3 + 2 * (ind % 5);
		kernel = cv::Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
		std::cout << "kernel_size: " << kernel_size << std::endl;
		std::cout << kernel << std::endl;

		// 使用滤波器
		cv::filter2D(src, dst, ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);
		imshow(window_name, dst);
		ind++;
	}

	return 0;
}
*/
/*
// 卷积研究
// 这里：https://blog.csdn.net/xvshu/article/details/81302441
// 有关于卷积的解释
// 看下面blur 的两次输出结果，配合手算更容易明白
int main()
{
	cv::Mat raw, dstBlur, dstBox;
	cv::Mat gb33, gb55, gb55_2, gb99, gb99_2;

	cv::Size size33(3, 3);
	cv::Point pt11n(-1, -1);
	cv::Point pt00(0, 0);

	raw = cv::Mat::zeros(100, 100, CV_8UC1);

	//raw.at<uchar>(50, 50) = 255;
	raw.at<uchar>(50, 50) = 10;

	//cv::blur(raw, dstBlur, size33);
	//cv::blur(raw, dstBlur, size33, pt00); // 移动锚点到左上角
	//std::cout << "dstBlur sum: " << cv::sum(dstBlur) << " dstBlur_kernel\n" << dstBlur.rowRange(45, 55).colRange(45, 55) << std::endl;

	//cv::boxFilter(raw, dstBox, -1, size33); // 使用默认参数，与blur 效果相同
	cv::boxFilter(raw, dstBox, -1, size33, pt11n, false); // 不使用归一化，中心点设较小的值看效果
	std::cout << "dstBox sum: " << cv::sum(dstBox) << " dstBox_kernel\n" << dstBox.rowRange(45, 55).colRange(45, 55) << std::endl;

	//cv::blur(dstBlur, dstBlur, size33);
	//cv::blur(dstBlur, dstBlur, size33, pt00); // 移动锚点到左上角
	//std::cout << "dstBlur sum: " << cv::sum(dstBlur) << " dstBlur_kernel\n" << dstBlur.rowRange(45, 55).colRange(45, 55) << std::endl;

	cv::boxFilter(dstBox, dstBox, -1, size33, pt11n, false); // 不使用归一化，中心点设较小的值看效果
	std::cout << "dstBox sum: " << cv::sum(dstBox) << " dstBox_kernel\n" << dstBox.rowRange(45, 55).colRange(45, 55) << std::endl;
	
	cv::imshow("raw", raw);
	cv::imshow("dstBox", dstBox);
	
	cv::waitKey(0);

	return 0;
}
*/
/*
// 观察不同过滤器处理效果
int main()
{
	cv::Mat raw;
	cv::Mat dstBlur, dstBox, dstBox2, dstMed, dstGauss, dstBila, dstBila2;
	cv::Mat dstSobel, dstScharr, dstLap;
	cv::Size size33(3, 3);
	cv::Point pt11n(-1, -1);

	raw = cv::imread("1.bmp");

	cv::blur(raw, dstBlur, size33);
	//cv::boxFilter(raw, dstBox, -1, size33);
	//cv::boxFilter(raw, dstBox2, -1, size33, pt11n, false);
	//cv::medianBlur(raw, dstMed, 3);
	//cv::GaussianBlur(raw, dstGauss, size33, 0);
	//cv::bilateralFilter(raw, dstBila, 5, 9, 9);
	//cv::bilateralFilter(raw, dstBila2, 9, 150, 150);
	cv::Sobel(raw, dstSobel, CV_8U, 1, 2);
	cv::Scharr(raw, dstScharr, CV_8U, 1, 0);
	cv::Laplacian(raw, dstLap, CV_8U);

	cv::imshow("raw", raw);
	//cv::imshow("blur", dstBlur);
	//cv::imshow("box default", dstBox);
	//cv::imshow("box non-normalize", dstBox2);
	//cv::imshow("median", dstMed);
	//cv::imshow("gauss", dstGauss);
	//cv::imshow("bilateral", dstBila);
	//cv::imshow("bilateral2", dstBila2);
	cv::imshow("sobel", dstSobel);
	cv::imshow("scharr", dstScharr);
	cv::imshow("laplacian", dstLap);

	cv::waitKey(0);

	return 0;
}
*/
/*
// 一种不均匀光照的补偿方法
// 参考：https://blog.csdn.net/hust_bochu_xuchao/article/details/54019994
using namespace cv;
void unevenLightCompensate(Mat& image, int blockSize)
{
	if (image.channels() == 3) cvtColor(image, image, 7);

	double average = mean(image)[0];

	int rows_new = ceil(double(image.rows) / double(blockSize));

	int cols_new = ceil(double(image.cols) / double(blockSize));

	Mat blockImage;

	blockImage = Mat::zeros(rows_new, cols_new, CV_32FC1);

	for (int i = 0; i < rows_new; i++) {

		for (int j = 0; j < cols_new; j++) {

			int rowmin = i * blockSize;

			int rowmax = (i + 1) * blockSize;

			if (rowmax > image.rows) rowmax = image.rows;

			int colmin = j * blockSize;

			int colmax = (j + 1) * blockSize;

			if (colmax > image.cols) colmax = image.cols;

			Mat imageROI = image(Range(rowmin, rowmax), Range(colmin, colmax));

			double temaver = mean(imageROI)[0];

			blockImage.at<float>(i, j) = temaver;

		}

	}

	blockImage = blockImage - average;

	Mat blockImage2;

	resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), INTER_CUBIC);

	Mat image2;

	image.convertTo(image2, CV_32FC1);

	Mat dst = image2 - blockImage2;

	dst.convertTo(image, CV_8UC1);
}

int main()
{
	cv::Mat raw;
	cv::Mat dst;
	
	raw = cv::imread("10-6-3.jpg");
	raw.copyTo(dst);

	unevenLightCompensate(dst, 32);

	cv::imshow("raw", raw);
	cv::imshow("dst", dst);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-6
// 参考：https://www.cnblogs.com/jukan/p/7815722.html
// 拉普拉斯算子过滤
// 只对较暗图像有效，如果太亮则不适用
int main()
{
	cv::Mat raw;
	cv::Mat dstBlur, dstBox, dstBox2, dstMed, dstGauss, dstBila, dstBila2;
	cv::Mat dstSobel, dstScharr, dstLap;
	cv::Size size33(3, 3);
	cv::Point pt11n(-1, -1);

	raw = cv::imread("10-6-4.jpg");

	cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, 1, 0, 0, -5, 0, 0, 1, 0);

	cv::filter2D(raw, dstLap, CV_8UC3, kernel);

	cv::imshow("raw", raw);
	cv::imshow("laplacian", dstLap);

	cv::waitKey(0);

	return 0;
}
*/
/*
// 同以上博客之gamma 校正
using namespace cv;

int main(int argc, char* argv[])

{

	Mat image = imread("10-6-6.jpg");

	Mat imageGamma(image.size(), CV_32FC3);

	for (int i = 0; i < image.rows; i++)

	{

		for (int j = 0; j < image.cols; j++)

		{

			imageGamma.at<Vec3f>(i, j)[0] = (image.at<Vec3b>(i, j)[0]) * (image.at<Vec3b>(i, j)[0]) * (image.at<Vec3b>(i, j)[0]);

			imageGamma.at<Vec3f>(i, j)[1] = (image.at<Vec3b>(i, j)[1]) * (image.at<Vec3b>(i, j)[1]) * (image.at<Vec3b>(i, j)[1]);

			imageGamma.at<Vec3f>(i, j)[2] = (image.at<Vec3b>(i, j)[2]) * (image.at<Vec3b>(i, j)[2]) * (image.at<Vec3b>(i, j)[2]);

		}

	}

	//归一化到0~255  

	normalize(imageGamma, imageGamma, 0, 255, NORM_MINMAX);

	//转换成8bit图像显示  

	convertScaleAbs(imageGamma, imageGamma);

	imshow("raw", image);

	imshow("gamma", imageGamma);

	waitKey();

	return 0;

}
*/
/*
// Exercise 10-6
// 分析：比较亮的地方r+g+b 值肯定较大，且r、g、b 单值也较大
//       比较暗的地方r+g+b 值肯定较小，且r、g、b 单值也较小
// 获取平均亮度
// 以下代码本来想按区部对总体的亮度对比来决定是否调整，
// 偏暗用laplace 算子，偏亮用gamma 校正
// 但gamma 校正显然是把一幅图像作为一个整体来考虑并处理的
// laplace 算子也类似，
// 所以作为一种可行方案，可以先判断整体是偏亮还是偏暗
// 然后单独使用一种方案，
// 或可以按较大的子块再进一步优化，
// 真有需求再做吧，第6 题到此为止
void lightingBlur(cv::InputArray src, cv::OutputArray dst)
{
}

int main()
{
	uchar r, g, b;
	double rGamma, gGamma, bGamma;
	double dGamaTemp = 0;
	int blockSize = 400;
	int lumiRaw, lumiRoi;

	cv::Scalar rawMean, roiMean;
	cv::Mat raw, roi, dst, gamma, roiDst, roiGamma;

	cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);

	raw = cv::imread("10-6-6.jpg");
	//raw = cv::imread("1.bmp");
	raw.copyTo(dst);

	gamma = cv::Mat(raw.size(), CV_32FC3);

	rawMean = cv::mean(raw);
	lumiRaw = rawMean[0] + rawMean[1] + rawMean[2];

	// 以blockSize 为界，逐块分析亮度，
	// 低则做Laplacian 过滤，高则做gama 校正
	int rowsOutline = raw.rows / blockSize;
	int colsOutLine = raw.cols / blockSize;

	for (int i = 0; i < rowsOutline; i++) {
		for (int j = 0; j < colsOutLine; j++) {
			roi = raw.rowRange(i * blockSize, (i + 1) * blockSize).colRange(j * blockSize, (j + 1) * blockSize);
			roiDst = dst.rowRange(i * blockSize, (i + 1) * blockSize).colRange(j * blockSize, (j + 1) * blockSize);
			roiGamma = gamma.rowRange(i * blockSize, (i + 1) * blockSize).colRange(j * blockSize, (j + 1) * blockSize);
			roiMean = cv::mean(roi);
			lumiRoi = roiMean[0] + roiMean[1] + roiMean[2];
			if (false && lumiRoi < lumiRaw * 0.8) {
				//cv::filter2D(roi, roiDst, CV_8UC3, kernel);
			}
			else if (true || lumiRoi > lumiRaw * 1.0) {
				for (int rIndex = 0; rIndex < blockSize; rIndex++) {
					for (int cIndex = 0; cIndex < blockSize; cIndex++) {
						b = roi.at<cv::Vec3b>(rIndex, cIndex)[0];
						g = roi.at<cv::Vec3b>(rIndex, cIndex)[1];
						r = roi.at<cv::Vec3b>(rIndex, cIndex)[2];

						bGamma = ((int)b) * ((int)b) * ((int)b);
						gGamma = ((int)g) * ((int)g) * ((int)g);
						rGamma = ((int)r) * ((int)r) * ((int)r);
						roiGamma.at<cv::Vec3f>(rIndex, cIndex)[0] = bGamma;
						roiGamma.at<cv::Vec3f>(rIndex, cIndex)[1] = gGamma;
						roiGamma.at<cv::Vec3f>(rIndex, cIndex)[2] = rGamma;
					}
				}

				cv::normalize(roiGamma, roiGamma, 0, 255, cv::NORM_MINMAX);
				cv::convertScaleAbs(roiGamma, roiGamma);
			}
		}
	}

	cv::imshow("raw", raw);
	cv::imshow("balance", roiGamma);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-7
// 总体感觉与第6 题类似，
// 对于天空，对应以blue  分量为主且较大的区域
// 对于湖面，对应以green 分量为主且较大的区域
int main()
{
	uchar r, g, b;
	int blockSize = 32;
	int bBlurCount = 0;
	int gBlurCount = 0;
	
	cv::Size size33(3, 3);
	cv::Size size99(9, 9);
	cv::Scalar roiMean;
	cv::Mat raw, roi, dst, roiDst;

	cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);

	//raw = cv::imread("10-6-1.jpg");
	//raw = cv::imread("sky2.jpg");
	raw = cv::imread("lake1.jpg");
	//raw = cv::imread("1.bmp");
	raw.copyTo(dst);

	// 以blockSize 为界，逐块分析亮度，
	// 低则做Laplacian 过滤，高则做gama 校正
	int rowsOutline = raw.rows / blockSize;
	int colsOutLine = raw.cols / blockSize;

	for (int i = 0; i < rowsOutline; i++) {
		for (int j = 0; j < colsOutLine; j++) {
			roi = raw.rowRange(i * blockSize, (i + 1) * blockSize).colRange(j * blockSize, (j + 1) * blockSize);
			roiDst = dst.rowRange(i * blockSize, (i + 1) * blockSize).colRange(j * blockSize, (j + 1) * blockSize);
			
			roiMean = cv::mean(roi);
			
			//if (roiMean[0] > 200 && roiMean[1] < 100 && roiMean[2] < 100) {
			if (roiMean[0] > 200) {
				cv::GaussianBlur(roi, roiDst, size99, 1.);
				bBlurCount++;
			}
			//else if (roiMean[1] > 200 && roiMean[2] < 100 && roiMean[0] < 100) {
			else if (roiMean[1] > 200) {
				cv::GaussianBlur(roi, roiDst, size99, 1.);
				gBlurCount++;
			}
		}
	}

	std::cout << "bBlurCount: " << bBlurCount << std::endl;
	std::cout << "gBlurCount: " << gBlurCount << std::endl;

	cv::imshow("raw", raw);
	//cv::imshow("sky", roiDst);
	cv::imshow("sky", dst);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-8
int main()
{
	int area, largest;
	cv::Scalar gray255(255);
	cv::Scalar gray100(100);
	cv::Scalar gray0(0);
	cv::Point ptSeed, ptLargest(0, 0);
	cv::Mat src1, src2, gray1, gray2;
	cv::Mat diff12, threshold50, opening, eroded, outline;
	cv::Mat cupMask;
	cv::Mat element, temp;
	src1 = cv::imread("10-8-1.jpg");
	src2 = cv::imread("10-8-2.jpg");

	cv::cvtColor(src1, gray1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(src2, gray2, cv::COLOR_BGR2GRAY);

	diff12 = cv::abs(gray1 - gray2);

	cv::threshold(diff12, threshold50, 50., 255., cv::THRESH_BINARY);

	cv::morphologyEx(threshold50, opening, cv::MORPH_OPEN, element);

	cv::morphologyEx(opening, eroded, cv::MORPH_ERODE, element);
	//cv::erode(opening, eroded, element);

	// 从左上角开始，如果像素值是255，使用100 进行漫水填充，
	// 如果此面积与之前的最大区域面积比较更大，则
	//   记录区域面积，记录种子点位置
	// 否则就把当前区域置为0（以0 漫水填充）
	eroded.copyTo(cupMask);
	largest = 0;
	for (int i = 0; i < cupMask.rows; i++) {
		for (int j = 0; j < cupMask.cols; j++) {
			if (255 == cupMask.at<uchar>(i, j)) {
				ptSeed.x = j;
				ptSeed.y = i;
				area = cv::floodFill(cupMask, ptSeed, gray100);
				if (area > largest) {
					cv::floodFill(cupMask, ptLargest, gray0);
					largest = area;
					ptLargest = ptSeed;
				}
				else {
					area = cv::floodFill(cupMask, ptSeed, gray0);
				}
				cv::imshow("cup mask", cupMask);
				cv::waitKey(1000);
			}
		}
	}
	cv::floodFill(cupMask, ptLargest, gray255);
	cv::imshow("cup mask", cupMask);

	//cv::imshow("gray2", gray2);
	//cv::imshow("diff12", diff12);
	//cv::imshow("threshold50", threshold50);
	//cv::imshow("opening", opening);
	//cv::imshow("eroded", eroded);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-9
int main()
{
	int area, largest;
	cv::Scalar gray255(255);
	cv::Scalar gray100(100);
	cv::Scalar gray0(0);
	cv::Point ptSeed, ptLargest(0, 0);
	cv::Mat src1, src2, gray1, gray2, outdoor;
	cv::Mat diff12, threshold50, opening, eroded, outline;
	cv::Mat cupMask;
	cv::Mat element, temp;
	src1 = cv::imread("10-8-1.jpg");
	src2 = cv::imread("10-8-2.jpg");
	
	outdoor = cv::imread("10-9.bmp");
	
	cv::cvtColor(src1, gray1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(src2, gray2, cv::COLOR_BGR2GRAY);

	diff12 = cv::abs(gray1 - gray2);

	cv::threshold(diff12, threshold50, 50., 255., cv::THRESH_BINARY);

	cv::morphologyEx(threshold50, opening, cv::MORPH_OPEN, element);

	cv::morphologyEx(opening, eroded, cv::MORPH_ERODE, element);
	//cv::erode(opening, eroded, element);

	// 从左上角开始，如果像素值是255，使用100 进行漫水填充，
	// 如果此面积与之前的最大区域面积比较更大，则
	//   记录区域面积，记录种子点位置
	// 否则就把当前区域置为0（以0 漫水填充）
	eroded.copyTo(cupMask);
	largest = 0;
	for (int i = 0; i < cupMask.rows; i++) {
		for (int j = 0; j < cupMask.cols; j++) {
			if (255 == cupMask.at<uchar>(i, j)) {
				ptSeed.x = j;
				ptSeed.y = i;
				area = cv::floodFill(cupMask, ptSeed, gray100);
				if (area > largest) {
					cv::floodFill(cupMask, ptLargest, gray0);
					largest = area;
					ptLargest = ptSeed;
				}
				else {
					area = cv::floodFill(cupMask, ptSeed, gray0);
				}
				cv::imshow("cup mask", cupMask);
				//cv::waitKey(1000);
			}
		}
	}
	cv::floodFill(cupMask, ptLargest, gray255);
	cv::imshow("cup mask", cupMask);

	// 尺寸相同的图片copyTo 进行掩膜可以成功
	// 尺寸不同的目标图片会被复制成源图片
	cv::copyTo(src2, outdoor, cupMask);
	cv::imshow("outdoor", outdoor);

	//cv::imshow("gray2", gray2);
	//cv::imshow("diff12", diff12);
	//cv::imshow("threshold50", threshold50);
	//cv::imshow("opening", opening);
	//cv::imshow("eroded", eroded);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-10
// 没看出玄机 :-(
int main()
{
	cv::Mat pic300, dst;
	cv::RNG rng;
	
	//pic300 = cv::Mat::zeros(300, 300, CV_8UC3);
	//rng.fill(pic300, cv::RNG::NORMAL, 0, 3);
	//cv::imshow("pic300", pic300);
	//cv::imwrite("pic300.jpg", pic300);
	
	
	pic300 = cv::imread("pic300.jpg");
	cv::bilateralFilter(pic300, dst, 5, 10, 150);
	cv::imshow("dst", dst);
	cv::imshow("pic300", pic300);
	
	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-11
int main()
{
	cv::Mat src, gray, tophat, kernel, mask, comp;
	cv::Mat gray2;
	
	//src = cv::imread("10-5-1.jpg");
	src = cv::imread("10-9.bmp");
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	cv::morphologyEx(gray, tophat, cv::MORPH_TOPHAT, kernel);
	cv::threshold(tophat, mask, 50, 255, cv::THRESH_BINARY);
	
	cv::imshow("src", src);
	cv::imshow("gray", gray);
	cv::imshow("tophat", tophat);
	cv::imshow("mask", mask);
	
	cv::cvtColor(gray, gray2, cv::COLOR_GRAY2BGR);
	//gray2.copyTo(src, mask);
	cv::copyTo(gray2, src, mask);
	cv::imshow("src-mask", src);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-12
// resize 的更清楚，pyrdown 较模糊
// resize 是插值，pyrdown 是滤波
int main()
{
	cv::Size size00(0, 0);
	cv::Mat src, r1, r2, r3, p1, p2, p3;
	
	//src = cv::imread("10-5-1.jpg");
	src = cv::imread("10-9.bmp");
	
	cv::resize(src, r1, size00, 0.5, 0.5);
	cv::resize(r1, r2, size00, 0.5, 0.5);
	cv::resize(r2, r3, size00, 0.5, 0.5);

	cv::pyrDown(src, p1);
	cv::pyrDown(p1, p2);
	cv::pyrDown(p2, p3);

	cv::imshow("r3", r3);
	cv::imshow("p3", p3);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-13
int main()
{
	int adaptiveMethod, blockSize, C;

	cv::Size size00(0, 0);
	cv::Mat src, src_8uc1;
	cv::Mat binary, binary_inv, trunc, tozero, tozero_inv;
	cv::Mat at_binary, at_binary_inv;

	src = cv::imread("1.bmp");
	//src = cv::imread("10-9.bmp");

	cv::cvtColor(src, src_8uc1, cv::COLOR_BGR2GRAY);

	//cv::threshold(src, binary, 128, 255, cv::THRESH_BINARY);
	//cv::threshold(src, binary_inv, 128, 255, cv::THRESH_BINARY_INV);
	//cv::threshold(src, trunc, 128, 255, cv::THRESH_TRUNC);
	//cv::threshold(src, tozero, 128, 255, cv::THRESH_TOZERO);
	//cv::threshold(src, tozero_inv, 128, 255, cv::THRESH_TOZERO_INV);

	//adaptiveMethod = cv::ADAPTIVE_THRESH_MEAN_C;
	adaptiveMethod = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
	//blockSize = 15;
	//blockSize = 5;
	blockSize = 3;
	//C = 10;
	//C = 5;
	//C = 0;
	C = -5;
	cv::adaptiveThreshold(src_8uc1, at_binary, 255, adaptiveMethod, cv::THRESH_BINARY, blockSize, C);
	cv::adaptiveThreshold(src_8uc1, at_binary_inv, 255, adaptiveMethod, cv::THRESH_BINARY_INV, blockSize, C);
	
	cv::imshow("src", src);
	//cv::imshow("binary", binary);
	//cv::imshow("binary_inv", binary_inv);
	//cv::imshow("trunc", trunc);
	//cv::imshow("tozero", tozero);
	//cv::imshow("tozero_inv", tozero_inv);
	
	cv::imshow("src_8uc1", src_8uc1);
	cv::imshow("at_binary", at_binary);
	cv::imshow("at_binary_inv", at_binary_inv);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-14
int main()
{
	cv::Mat src, dst;
	
	src = cv::imread("1.bmp");
	cv::pyrMeanShiftFiltering(src, dst, 10, 10);
	cv::imshow("src", src);
	cv::imshow("dst", dst);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-14
// 参考：😊😂🤣❤
// https://answers.opencv.org/question/197167/how-to-detect-only-60-degree-lines-a-question-from-learning-opencv3-chapter-10/
using namespace cv;
int main()
{
	Mat dst;
	Mat srcLines(100, 100, CV_8UC1, Scalar::all(0));//draw lines
	line(srcLines, Point(0, 0), Point(100, 100), Scalar(255), 1);
	line(srcLines, Point(50, 50), Point(100, 0), Scalar(255), 1);//45 - degree
	line(srcLines, Point(0, 100), Point(50, 0), Scalar(255), 1);//60 - degree
	Mat srcH(3, 3, CV_8UC1, Scalar::all(0));

	srcH.at<char>(0, 0) = 255;
	srcH.at<char>(1, 1) = 255;
	srcH.at<char>(2, 2) = 255;

	filter2D(srcLines, dst, srcLines.depth(), srcH);

	cv::imshow("src", srcLines);
	cv::imshow("dst", dst);

	cv::waitKey(0);

	return 0;
}
*/
/*
// 同上
using namespace cv;
int main()
{
	Mat src, dst;
	
	cv::Mat1f k = (cv::Mat1f(9, 9) <<
		0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0);

	Mat srcH(3, 3, CV_8UC1, Scalar::all(0));

	srcH.at<uchar>(0, 0) = 255;
	srcH.at<uchar>(1, 1) = 255;
	srcH.at<uchar>(2, 2) = 255;

	src = cv::imread("1.bmp");

	//filter2D(src, dst, src.depth(), srcH);
	filter2D(src, dst, src.depth(), k);

	cv::imshow("src", src);
	cv::imshow("dst", dst);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-16
int main()
{
	cv::Mat src, src_32f, dst, gray;
	cv::Mat1f kernel, rowKernel, colKernel;
	cv::Mat1f kernel2, rowKernel2, colKernel2;
	
	kernel = (cv::Mat1f(3, 3) <<
		1/16, 2/16, 1/16,
		2/16, 4/16, 2/16,
		1/16, 2/16, 1/16);

	rowKernel = (cv::Mat1f(1, 3) <<
		1 / 16, 1 / 16, 1 / 16);
		//1, 2, 1);

	colKernel = (cv::Mat1f(3, 1) <<
		1, 2, 1);
		//1 / 16, 1 / 16, 1 / 16);

	rowKernel2 = (cv::Mat1f(1, 3) <<
		1 / 4, 2 / 4, 1 / 4);

	colKernel2 = (cv::Mat1f(3, 1) <<
		1/4, 2/4, 1/4);

	int ddepth;
	
	src = cv::imread("1.bmp");
	if (src.empty()) {
		return -1;
	}

	//cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

	ddepth = -1;
	//cv::sepFilter2D(gray, dst, ddepth, rowKernel, colKernel);
	//cv::sepFilter2D(src, dst, ddepth, rowKernel, colKernel);

	src.convertTo(src_32f, CV_32FC3, 1 / 255.0);
	cv::filter2D(src_32f, dst, -1, colKernel2);
	//cv::filter2D(src_32f, dst, -1, rowKernel2);
	//cv::filter2D(src_32f, dst, -1, kernel);
	
	imshow("src", src);
	imshow("src_32f", src_32f);
	//imshow("gray", gray);
	imshow("dst", dst);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-17
int main()
{
	cv::Mat src, dstK, dstKrc, dstKcr, dstScharr;
	cv::Mat1f kernel, rowKernel, colKernel;
	cv::Mat1f kernel2, rowKernel2, colKernel2;

	kernel = (cv::Mat1f(3, 3) <<
		-3, 0, 3,
		-10, 0, 10,
		-3, 0, 3);

	rowKernel = (cv::Mat1f(1, 3) <<
		-1, 0, 1);

	colKernel = (cv::Mat1f(3, 1) <<
		3, 10, 3);

	int ddepth;

	src = cv::imread("1.bmp");
	if (src.empty()) {
		return -1;
	}

	ddepth = -1;
	cv::sepFilter2D(src, dstK, ddepth, rowKernel, colKernel);

	cv::filter2D(src, dstKrc, ddepth, rowKernel);
	cv::filter2D(dstKrc, dstKrc, ddepth, colKernel);

	cv::filter2D(src, dstKcr, ddepth, colKernel);
	cv::filter2D(dstKcr, dstKcr, ddepth, rowKernel);

	//cv::Sobel(src, dstScharr, ddepth, 0, 1, cv::FILTER_SCHARR); // 顺序0 1 与上面不一样
	cv::Sobel(src, dstScharr, ddepth, 1, 0, cv::FILTER_SCHARR); // 顺序1 0 这个与上面的结果一样

	imshow("src", src);
	imshow("dstK", dstK);
	imshow("dstKrc", dstKrc);
	imshow("dstKcr", dstKcr);
	imshow("dstScharr", dstScharr);

	cv::waitKey(0);

	return 0;
}
*/

// Exercise 10-18
// 窗口越大，线条越粗 :-)
int main()
{
	int ddepth = -1;
	cv::Mat src, dst33, dst55, dst99, dst1313;
	
	src = cv::imread("10-18.png");
	if (src.empty()) {
		return -1;
	}

	cv::Sobel(src, dst33, ddepth, 0, 1, 3);
	cv::Sobel(src, dst55, ddepth, 0, 1, 5);
	cv::Sobel(src, dst99, ddepth, 0, 1, 9);
	cv::Sobel(src, dst1313, ddepth, 0, 1, 13);

	imshow("src", src);
	imshow("dst33", dst33);
	imshow("dst55", dst55);
	imshow("dst99", dst99);
	imshow("dst1313", dst1313);

	cv::waitKey(0);

	return 0;
}

/*
// Exercise 10-19
// 保留
int main()
{
	int ddepth = -1;
	cv::Mat dst33, dst55, dst99;
	cv::Mat dstGradient, kernel;
	cv::Mat src(300, 300, CV_8UC1, cv::Scalar::all(0));//draw lines
	cv::line(src, cv::Point(300, 0), cv::Point(0, 300), cv::Scalar(255), 1);

	cv::morphologyEx(src, dstGradient, cv::MORPH_GRADIENT, kernel);
	
	cv::Sobel(src, dst33, ddepth, 0, 1, 3);
	cv::Sobel(src, dst55, ddepth, 0, 1, 5);
	cv::Sobel(src, dst99, ddepth, 0, 1, 9);

	cv::imshow("src", src);
	cv::imshow("dst_gradient", dstGradient);
	cv::imshow("dst33", dst33);
	cv::imshow("dst55", dst55);
	cv::imshow("dst99", dst99);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 10-19
// 参考：https://answers.opencv.org/question/211641/calculating-a-magnitude/
// 保留
int main()
{
	int ksize;

	cv::UMat gr;
	cv::UMat dx, dy, dxs, dys;

	cv::Mat src(300, 300, CV_8UC1, cv::Scalar::all(0));//draw lines
	cv::line(src, cv::Point(300, 0), cv::Point(0, 300), cv::Scalar(255), 1);
	
	src.copyTo(gr);

	//ksize = 3;
	//ksize = 5;
	ksize = 9;
	cv::Sobel(gr, dx, CV_32F, 1, 0, ksize);
	cv::Sobel(gr, dy, CV_32F, 0, 1, ksize);

	cv::resize(dx, dxs, cv::Size(dx.rows, dx.rows), 0, 0, cv::INTER_AREA);
	cv::resize(dy, dys, cv::Size(dx.rows, dx.rows), 0, 0, cv::INTER_AREA);

	cv::UMat mag;

	cv::magnitude(dxs, dys, mag);

	cv::Mat res;
	cv::hconcat(dxs, dys, res);
	cv::hconcat(res, mag, res);

	cv::imshow("res", res);

	cv::waitKey(0);

	return 0;
}
*/