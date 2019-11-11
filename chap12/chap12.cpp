// chap12.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/*
// Example 12-1
int main(int argc, char **argv)
{
	if (argc != 2) {
		std::cout << "Fourier Transform\nUsage: " 
			<< argv[0] << "<imagename>" << std::endl;
		return -1;
	}

	cv::Mat A = cv::imread(argv[1], 0);

	if (A.empty()) {
		std::cout << "Cannot load " << argv[1] << std::endl;
		return -1;
	}

	cv::Size patchSize(100, 100);
	cv::Point topleft(A.cols / 2, A.rows / 2);
	cv::Rect roi(topleft.x, topleft.y, patchSize.width, patchSize.height);
	cv::Mat B = A(roi);

	int dft_M = cv::getOptimalDFTSize(A.rows + B.rows - 1);
	int dft_N = cv::getOptimalDFTSize(A.cols + B.cols - 1);

	cv::Mat dft_A = cv::Mat::zeros(dft_M, dft_N, CV_32F);
	cv::Mat dft_B = cv::Mat::zeros(dft_M, dft_N, CV_32F);

	cv::Mat dft_A_part = dft_A(cv::Rect(0, 0, A.cols, A.rows));
	cv::Mat dft_B_part = dft_B(cv::Rect(0, 0, B.cols, B.rows));

	A.convertTo(dft_A_part, dft_A_part.type(), 1, -cv::mean(A)[0]);
	B.convertTo(dft_B_part, dft_B_part.type(), 1, -cv::mean(B)[0]);

	cv::dft(dft_A, dft_A, 0, A.rows);
	cv::dft(dft_B, dft_B, 0, B.rows);

	cv::mulSpectrums(dft_A, dft_B, dft_A, 0, true);
	cv::idft(dft_A, dft_A, cv::DFT_SCALE, A.rows + B.rows - 1);

	cv::Mat corr = dft_A(cv::Rect(0, 0, A.cols + B.cols - 1, A.rows + B.rows - 1));
	cv::normalize(corr, corr, 0, 1, cv::NORM_MINMAX, corr.type());
	cv::pow(corr, 3., corr);

	//cv::B ^= cv::Scalar::all(255);
	cv::imshow("Image", A);
	cv::imshow("Correlation", corr);

	cv::waitKey();
	return 0;
}
*/
/*
// Example 12-2
int main(int argc, char** argv)
{
	if (argc != 2) {
		std::cout << "Hough Circle detect\nUsage: "
			<< argv[0] << "<imagename>" << std::endl;
		return -1;
	}

	cv::Scalar RED(0, 0, 255);
	cv::Mat src, image;
	src = cv::imread(argv[1], 1);

	if (src.empty()) {
		std::cout << "Cannot load " << argv[1] << std::endl;
		return -1;
	}

	cv::cvtColor(src, image, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(image, image, cv::Size(5, 5), 0, 0);

	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(image, circles, cv::HOUGH_GRADIENT, 2, image.cols / 10);

	for (size_t i = 0; i < circles.size(); i++) {
		cv::circle(src, cv::Point(cvRound(circles[i][0]), 
			cvRound(circles[i][1])), cvRound(circles[i][2]), 
			RED, 2, cv::LINE_AA);
	}

	cv::imshow("image", image);
	cv::imshow("Hough Circles", src);
	
	cv::waitKey();
	return 0;
}
*/
/*
// Exercise 12-1
// 高值越小保留的细节越多
// 比例越高保留的细节越多
// 保留细节越多时耗时也越多
// 由上到下时间由0.19s 逐步降低到0.052s
int main(int argc, char** argv) {
	std::cout << argv[1] << std::endl;

	cv::Mat img_rgb, img_gry, img_cny15, img_cny275,img_cny41;

	img_rgb = cv::imread(argv[1], -1);
	if (img_rgb.empty()) {
		return -1;
	}

	cv::imshow("raw", img_rgb);

	cv::cvtColor(img_rgb, img_gry, cv::COLOR_BGR2GRAY);
	cv::imshow("gray", img_gry);

	double dFrq = cv::getTickFrequency();
	int64 i64Begin = cv::getTickCount();

	// high < 50
	//cv::Canny(img_gry, img_cny15, 18, 28, 3, true); // 1.5:1 or 3:2
	//cv::Canny(img_gry, img_cny275, 10, 28, 3, true); // 2.75:1 or 5.5:2 or 11:4
	//cv::Canny(img_gry, img_cny41, 7, 28, 3, true); // 4:1

	// high 50 ~ 100
	//cv::Canny(img_gry, img_cny15, 40, 60, 3, true); // 1.5:1 or 3:2
	//cv::Canny(img_gry, img_cny275, 22, 60, 3, true); // 2.75:1 or 5.5:2 or 11:4
	//cv::Canny(img_gry, img_cny41, 15, 60, 3, true); // 4:1

	// high 100 ~ 150
	//cv::Canny(img_gry, img_cny15, 80, 120, 3, true); // 1.5:1 or 3:2
	//cv::Canny(img_gry, img_cny275, 44, 120, 3, true); // 2.75:1 or 5.5:2 or 11:4
	//cv::Canny(img_gry, img_cny41, 30, 120, 3, true); // 4:1

	// high 150 ~ 200
	//cv::Canny(img_gry, img_cny15, 120, 180, 3, true); // 1.5:1 or 3:2
	//cv::Canny(img_gry, img_cny275, 65, 180, 3, true); // 2.75:1 or 5.5:2 or 11:4
	//cv::Canny(img_gry, img_cny41, 45, 180, 3, true); // 4:1

	// high 200 ~ 250
	cv::Canny(img_gry, img_cny15, 147, 220, 3, true); // 1.5:1 or 3:2
	cv::Canny(img_gry, img_cny275, 80, 220, 3, true); // 2.75:1 or 5.5:2 or 11:4
	cv::Canny(img_gry, img_cny41, 55, 220, 3, true); // 4:1

	int64 i64Cur = cv::getTickCount();
	int64 i64TickCount = i64Cur - i64Begin;
	std::cout << "tick count: " << i64TickCount << std::endl;
	std::cout << "tick frequency: " << dFrq << std::endl;
	std::cout << "time(s): " << i64TickCount / dFrq << std::endl;

	cv::imshow("1.5:1", img_cny15);
	cv::imshow("2.75:1", img_cny275);
	cv::imshow("4:1", img_cny41);

	cv::waitKey(0);

	return 0;
}
*/
/*
// Exercise 12-2
int main(int argc, char** argv)
{
	if (argc != 2) {
		std::cout << "Hough Circle detect\nUsage: "
			<< argv[0] << "<imagename>" << std::endl;
		return -1;
	}

	cv::Scalar RED(0, 0, 255);
	cv::Mat src, image;
	src = cv::imread(argv[1], 1);

	if (src.empty()) {
		std::cout << "Cannot load " << argv[1] << std::endl;
		return -1;
	}

	cv::cvtColor(src, image, cv::COLOR_BGR2GRAY);
	//cv::GaussianBlur(image, image, cv::Size(5, 5), 0, 0);
	cv::Canny(image, image, 50, 175);
	
	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(image, circles, cv::HOUGH_GRADIENT, 1, image.cols / 10);

	for (size_t i = 0; i < circles.size(); i++) {
		cv::circle(src, cv::Point(cvRound(circles[i][0]),
			cvRound(circles[i][1])), cvRound(circles[i][2]),
			RED, 1, cv::LINE_AA);
	}
	
	cv::Point pt1, pt2;
	std::vector<cv::Vec2f> lines;
	cv::HoughLines(image, lines, 20, CV_PI / 180*20, 100);
	for (size_t i = 0; i < lines.size(); i++) { 
		//将求得的线条画出来
		float rho = lines[i][0], theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		cv::line(src, pt1, pt2, RED, 1, cv::LINE_AA);
	}
	
	cv::imshow("image", image);
	cv::imshow("Hough Circles", src);

	cv::waitKey();
	return 0;
}
*/
// Exercise 12-3
// 由小线段和小弧线可以组合成任何形状
//
/*
// Exercise 12-4
int main()
{
	cv::Mat gauss, image;

	gauss = cv::Mat::zeros(100, 100, CV_8UC1);
	cv::randn(gauss, 128, 32);

	cv::imshow("normal_gauss", gauss);

	//image = cv::imread("290.png", 0);
	//image = cv::imread("2.jpg", 0);
	image = cv::imread("1.jpg", 0);
	//cv::imshow("image", image);


	int dft_M = cv::getOptimalDFTSize(image.rows + gauss.rows - 1);
	int dft_N = cv::getOptimalDFTSize(image.cols + gauss.cols - 1);

	cv::Mat dft_A = cv::Mat::zeros(dft_M, dft_N, CV_32F);
	cv::Mat dft_B = cv::Mat::zeros(dft_M, dft_N, CV_32F);

	cv::Mat dft_A_part = dft_A(cv::Rect(0, 0, image.cols, image.rows));
	cv::Mat dft_B_part = dft_B(cv::Rect(0, 0, gauss.cols, gauss.rows));

	image.convertTo(dft_A_part, dft_A_part.type(), 1, -cv::mean(image)[0]);
	gauss.convertTo(dft_B_part, dft_B_part.type(), 1, -cv::mean(gauss)[0]);

	cv::dft(dft_A, dft_A, 0, image.rows);
	cv::dft(dft_B, dft_B, 0, gauss.rows);

	cv::mulSpectrums(dft_A, dft_B, dft_A, 0, true);
	cv::idft(dft_A, dft_A, cv::DFT_SCALE, image.rows + gauss.rows - 1);

	cv::Mat corr = dft_A(cv::Rect(0, 0, image.cols + gauss.cols - 1, image.rows + gauss.rows - 1));
	cv::normalize(corr, corr, 0, 1, cv::NORM_MINMAX, corr.type());
	cv::pow(corr, 3., corr);

	//cv::B ^= cv::Scalar::all(255);
	cv::imshow("Image", image);
	cv::imshow("Correlation", corr);

	cv::waitKey();
	return 0;
}
*/
/*
// Exercise 12-5
int main()
{
	cv::Scalar WHITE(255, 255, 255);
	cv::Point pt1, pt2;
	cv::Mat src, binary, sum, dst;

	int ls_w, ls_h, ls_area; // long skinny rectangle's width height and area
	int ls_sum;
	int pix_sum; // long skinny rectangle's pix value sum
	double lt, rt, lb, rb, inte_val;
	int pix_val;

	src = cv::Mat::zeros(300, 300, CV_8UC1);

	//src = cv::imread("290.png", 0);
	//image = cv::imread("2.jpg", 0);
	src = cv::imread("2.jpg", cv::IMREAD_GRAYSCALE);

	dst = cv::Mat::zeros(src.size(), CV_8UC1);
	
	pt1.x = 100;
	pt1.y = 100;
	pt2.x = 200;
	pt2.y = 200;
	cv::rectangle(src, pt1, pt2, WHITE);

	cv::threshold(src, binary, 128, 255, cv::THRESH_BINARY);
	pix_val = 255;

	cv::integral(binary, sum, CV_64F);

	ls_w = 50;
	ls_h = 1;
	ls_area = ls_w * ls_h;
	ls_sum = ls_area * pix_val;
	
	// horizontal
	for (int i = 0; i < sum.rows - ls_h; i++) {
		for (int j = 0; j < sum.cols - ls_w; j++) {
			if (binary.at<uchar>(i, j) > 0) {
				lt = sum.at<double>(i, j);
				rt = sum.at<double>(i, j + ls_w);
				lb = sum.at<double>(i + ls_h, j);
				rb = sum.at<double>(i + ls_h, j + ls_w);
				inte_val = rb - lb - rt + lt;
				if (abs(inte_val - ls_sum) < 0.0001) {
					// found feature
					src.rowRange(i, i + ls_h).colRange(j, j + ls_w).copyTo(dst.rowRange(i, i + ls_h).colRange(j, j + ls_w));
				}
			}
			
		}
	}

	// vectical
	ls_w = 1;
	ls_h = 50;
	ls_area = ls_w * ls_h;
	ls_sum = ls_area * pix_val;
	for (int i = 0; i < sum.rows - ls_h; i++) {
		for (int j = 0; j < sum.cols - ls_w; j++) {
			if (binary.at<uchar>(i, j) > 0) {
				lt = sum.at<double>(i, j);
				rt = sum.at<double>(i, j + ls_w);
				lb = sum.at<double>(i + ls_h, j);
				rb = sum.at<double>(i + ls_h, j + ls_w);
				inte_val = rb - lb - rt + lt;
				if (abs(inte_val - ls_sum) < 0.0001) {
					// found feature
					src.rowRange(i, i + ls_h).colRange(j, j + ls_w).copyTo(dst.rowRange(i, i + ls_h).colRange(j, j + ls_w));
				}
			}

		}
	}

	cv::imshow("src", src);
	//cv::imshow("sum", sum);
	cv::imshow("binary", binary);
	cv::imshow("dst", dst);

	cv::waitKey();
	return 0;
}
*/
/*
// 积分图提取边缘
// 参考：https://cloud.tencent.com/developer/article/1084469
int getblockSum(cv::Mat& sum, int x1, int y1, int x2, int y2, int i)
{
	int tl = sum.at<cv::Vec3i>(y1, x1)[i];
	int tr = sum.at<cv::Vec3i>(y1, x2)[i];
	int bl = sum.at<cv::Vec3i>(y2, x1)[i];
	int br = sum.at<cv::Vec3i>(y2, x2)[i];
	int s = (br - bl - tr + tl);

	return s;
}

int main()
{
	cv::Scalar WHITE(255, 255, 255);
	cv::Point pt1, pt2;
	cv::Mat src, sum, result, dst, gray;

	src = cv::imread("290.png", 1);
	//image = cv::imread("2.jpg", 0);
	//src = cv::imread("2.jpg", cv::IMREAD_GRAYSCALE);

	cv::integral(src, sum, CV_32S);
	std::cout << "sum size: " << sum.size() << std::endl;
	std::cout << "sum channel: " << sum.channels() << std::endl;

	int w = src.cols;
	int h = src.rows;

	result = cv::Mat::zeros(src.size(), CV_32SC3);

	int x2 = 0, y2 = 0;
	int x1 = 0, y1 = 0;
	int ksize = 3; // 算子大小，可以修改，越大边缘效应越明显
	int radius = ksize / 2;
	int ch = src.channels();
	int cx = 0, cy = 0;
	int s1 = 0, s2 = 0;

	for (int row = 0; row < h + radius; row++) {
		y2 = (row + 1) > h ? h : (row + 1);
		y1 = (row - ksize) < 0 ? 0 : (row - ksize);

		for (int col = 0; col < w + radius; col++) {
			x2 = (col + 1) > w ? w : (col + 1);
			x1 = (col - ksize) < 0 ? 0 : (col - ksize);
			cx = (col - radius) < 0 ? 0 : (col - radius);
			cy = (row - radius) < 0 ? 0 : (row - radius);

			for (int i = 0; i < ch; i++) {
				//s1 = getblockSum(sum, x1, y1, cx, cy, i);
				//s2 = getblockSum(sum, cx, cy, x2, y2, i);
				s1 = getblockSum(sum, x1, y1, cx, y2, i);
				s2 = getblockSum(sum, cx, y1, x2, y2, i);
				result.at<cv::Vec3i>(cy, cx)[i] = cv::saturate_cast<int>(s2 - s1);
			}
		}
	}

	cv::convertScaleAbs(result, dst);
	cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
	cv::cvtColor(dst, gray, cv::COLOR_BGR2GRAY);

	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::imshow("gray", gray);

	cv::waitKey();
	return 0;
}
*/
/*
// Exercise 12-6
// 参考：https://wenku.baidu.com/view/4473d322bcd126fff7050bf9.html
//
// ----------------------------------------
// |       /\            .                 
// |      /  \          .  .              
// |     /    \        .     .            
// |    /      \      .        .           
// |    \      /        .     .          
// |     \    /           .  .            
// |      \  /              .             
// |       \/                             
// |                                       
// 从左上角(0,0)往右下方斜向45度积分
// 
// 2*2                 tilted               0   0   0
//               1  1                       0   1   1
//               1  1                       1   3   3
//
// 3*3                 tilted           0   0   0   0
//            1  1  1                   0   1   1   1
//            1  1  1                   1   3   4   3
//            1  1  1                   3   6   7   6
// 
// 4*4                 tilted       0   0   0   0   0
//         1  1  1  1               0   1   1   1   1
//         1  1  1  1               1   3   4   4   3
//         1  1  1  1               3   6   8   8   6 
//         1  1  1  1               6  10  12  12  10
//
// 5*5                 tilted   0   0   0   0   0   0
//      1  1  1  1  1           0   1   1   1   1   1
//      1  1  1  1  1           1   3   4   4   4   3
//      1  1  1  1  1           3   6   8   9   8   6 
//      1  1  1  1  1           6  10  13  14  13  10 
//      1  1  1  1  1          10  15  18  19  18  15
// 
// 如果不看别的，单以上面数据还真不是太好总结出算法
// 下面再看单位矩阵的
// 2*2                 tilted               0   0   0
//               1  0                       0   1   0
//               0  1                       1   1   2
//
// 3*3                 tilted           0   0   0   0
//            1  0  0                   0   1   0   0
//            0  1  0                   1   1   2   0
//            0  0  1                   1   2   2   3
// 
// 4*4                 tilted       0   0   0   0   0
//         1  0  0  0               0   1   0   0   0
//         0  1  0  0               1   1   2   0   0
//         0  0  1  0               1   2   2   3   0 
//         0  0  0  1               2   2   3   3   4
//
// 5*5                 tilted   0   0   0   0   0   0
//      1  0  0  0  0           0   1   0   0   0   0
//      0  1  0  0  0           1   1   2   0   0   0
//      0  0  1  0  0           1   2   2   3   0   0 
//      0  0  0  1  0           2   2   3   3   4   0 
//      0  0  0  0  1           2   3   3   4   4   5
//
// 通过在一个Excel 文档中标注每个点对应的积分和，分析出具体算法
// 
// 现在知道了数据怎么组成的，以当前点为首，以45度角逐层往上退
// 先写一个计算单个点旋转积分和的函数
// mat 是源矩阵
// x, y 是积分图的坐标，范围分别是源矩阵的宽高+1
// 注：不包括积分图的第一行和第一列
double calcSingleRotationIntegral(cv::Mat &mat, int x, int y)
{
	double inte_val = 0;

	if (x < 0 || y < 0 || x > mat.cols || y > mat.rows) {
		throw "calcSingleRotationIntegral param error.";
	}

	if (0 == y) {
		return 0;
	}

	if (0 == x && 1 == y) {
		return 0;
	}

	if (1 == y) {
		return mat.at<uchar>(x-1, y-1);
	}

	// 由(x, y) 逐层往上退
	inte_val = mat.at<uchar>(x - 1, y - 1);
	for (int i = y - 1; i > 0; i--) {
		int end = (((x + y - i) < mat.cols) ? (x + y - i) : mat.cols);
		for (int j = ((x - (y-i)) < 1 ? 1 : (x - (y-i))); j <= end; j++) {
			inte_val += mat.at<uchar>(i-1, j-1);
		}
	}

	return inte_val;
}

// 通用计算旋转积分图方法
// 注：受参考论文启发，估计OpenCV库采用了同样的算法，等待看源码后验证
// 
void calcRotationIntegral(cv::Mat& src, cv::Mat& dst)
{
	int ROW = src.rows;
	int COL = src.cols;
	int RIROW = ROW + 1;
	int RICOL = COL + 1;

	int ri_lt; // ri(x-1, y-1)
	int ri_rt; // ri(x+1, y-1)
	int ri_tt; // ri(x, y-2)
	int v_t; // v(x, y-1)
	int v; // v(x, y)

	// 使用CV_32S 便于小样本测试观察输出
	dst = cv::Mat::zeros(RIROW, RICOL, CV_32S);

	// 第一(0)行全0，从i = 1 开始
	for (int i = 1; i < RIROW; i++) {
		for (int j = 0; j < RICOL; j++) {
			ri_lt = ((j - 1 < 0) ? 0 : dst.at<int>(i - 1, j - 1));
			ri_rt = ((j + 1 > ROW) ? 0 : dst.at<int>(i - 1, j + 1));
			ri_tt = ( (0 == ri_lt || 0 == ri_rt || i-2 < 0)? 0:dst.at<int>(i-2,j) ); // 左侧或右侧没有值时就不需要减去重复的值
			v_t = (i-2<0 || j-1<0) ? 0 :src.at<uchar>(i-1-1, j-1);
			v = (i - 1 < 0 || j - 1 < 0) ? 0 : src.at<uchar>(i - 1, j - 1);
			dst.at<int>(i, j) = ri_lt + ri_rt - ri_tt + v_t + v;
		}
	}
}

int main()
{
	// 首先测试一下OpenCV 是怎样计算旋转的积分图
	const int ROW = 5;
	const int COL = 5;
	cv::Scalar WHITE(255, 255, 255);
	cv::Point pt1, pt2;
	cv::Mat src, sum, sqsum, tilted, calc_single, dst;

	src = cv::Mat::ones(5, 5, CV_8UC1);
	//src = cv::Mat::eye(ROW, COL, CV_8UC1);

	cv::integral(src, sum, sqsum, tilted, CV_32S);
	std::cout << "sum: \n" << sum << std::endl;
	std::cout << "sqsum: \n" << sqsum << std::endl;
	std::cout << "tilted: \n" << tilted << std::endl;

	int w = src.cols;
	int h = src.rows;

	// 单点计算方法
	std::cout << "size of int: " << sizeof(int) << std::endl;
	
	calc_single = cv::Mat::zeros(ROW+1, COL+1, CV_32S);
	std::cout << "calc_single.elemSize(): " << calc_single.elemSize() << std::endl;
	std::cout << "calc_single.elemSize1(): " << calc_single.elemSize1() << std::endl;
	for (int i = 1; i < ROW + 1; i++) {
		for (int j = 1; j < COL+1; j++) {
			calc_single.at<int>(i, j) = (int)calcSingleRotationIntegral(src, j, i);
		}
	}
	std::cout << "calc_single: \n" << calc_single << std::endl;

	calcRotationIntegral(src, dst);
	std::cout << "dst: \n" << dst << std::endl;

	cv::waitKey();

	return 0;
}
*/
/*
// Exercise 12-7
// 参考：http://www.cjig.cn/jig/ch/reader/view_abstract.aspx?file_no=20180105&flag=1
// 或 http://www.cjig.cn/html/jig/2018/1/20180105.htm
int main()
{
	cv::Mat src, gray, dst, binary;
	//src = cv::imread("1.jpg");
	src = cv::imread("2.jpg");
	//src = cv::imread("290.png");
	//src = cv::imread("12-7.png");
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	cv::threshold(gray, binary, 50, 255, cv::THRESH_BINARY);
	cv::distanceTransform(binary, dst, cv::DistanceTypes::DIST_C, 5);
	
	std::cout << "before normalize:\n" << dst.rowRange(100, 120).colRange(100, 110) << std::endl;
	cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
	std::cout << "after normalize:\n" << dst.rowRange(100, 120).colRange(100, 110) << std::endl;
	
	cv::imshow("src", src);
	cv::imshow("binary", binary);
	cv::imshow("gray", gray);
	cv::imshow("dst", dst);
	
	cv::waitKey();

	return 0;
}
*/
/*
// Exercise 12-8
int main()
{
	cv::Size size49(49, 49);
	cv::Mat src, gray, dst;
	src = cv::imread("bicycle.jpg");

	double dFrq = cv::getTickFrequency();
	int64 i64Begin = cv::getTickCount();
	cv::GaussianBlur(src, dst, size49, 0);
	int64 i64Cur = cv::getTickCount();
	int64 i64TickCount = i64Cur - i64Begin;
	std::cout << "tick count: " << i64TickCount << std::endl;
	std::cout << "tick frequency: " << dFrq << std::endl;
	std::cout << "time(s): " << i64TickCount / dFrq << std::endl;

	cv::imshow("src", src);
	cv::imshow("dst", dst);

	cv::waitKey();

	return 0;
}
*/
/*
// Exercise 12-8
// Example 12-1 加上时间监测
int main(int argc, char** argv)
{
	if (argc != 2) {
		std::cout << "Fourier Transform\nUsage: "
			<< argv[0] << "<imagename>" << std::endl;
		return -1;
	}

	cv::Mat A = cv::imread(argv[1], 0);

	if (A.empty()) {
		std::cout << "Cannot load " << argv[1] << std::endl;
		return -1;
	}

	double dFrq = cv::getTickFrequency();
	int64 i64Begin = cv::getTickCount();

	cv::Size patchSize(100, 100);
	cv::Point topleft(A.cols / 2, A.rows / 2);
	cv::Rect roi(topleft.x, topleft.y, patchSize.width, patchSize.height);
	cv::Mat B = A(roi);

	int dft_M = cv::getOptimalDFTSize(A.rows + B.rows - 1);
	int dft_N = cv::getOptimalDFTSize(A.cols + B.cols - 1);

	cv::Mat dft_A = cv::Mat::zeros(dft_M, dft_N, CV_32F);
	cv::Mat dft_B = cv::Mat::zeros(dft_M, dft_N, CV_32F);

	cv::Mat dft_A_part = dft_A(cv::Rect(0, 0, A.cols, A.rows));
	cv::Mat dft_B_part = dft_B(cv::Rect(0, 0, B.cols, B.rows));

	A.convertTo(dft_A_part, dft_A_part.type(), 1, -cv::mean(A)[0]);
	B.convertTo(dft_B_part, dft_B_part.type(), 1, -cv::mean(B)[0]);

	cv::dft(dft_A, dft_A, 0, A.rows);
	cv::dft(dft_B, dft_B, 0, B.rows);

	cv::mulSpectrums(dft_A, dft_B, dft_A, 0, true);
	cv::idft(dft_A, dft_A, cv::DFT_SCALE, A.rows + B.rows - 1);

	cv::Mat corr = dft_A(cv::Rect(0, 0, A.cols + B.cols - 1, A.rows + B.rows - 1));
	cv::normalize(corr, corr, 0, 1, cv::NORM_MINMAX, corr.type());
	cv::pow(corr, 3., corr);

	int64 i64Cur = cv::getTickCount();
	int64 i64TickCount = i64Cur - i64Begin;
	std::cout << "tick count: " << i64TickCount << std::endl;
	std::cout << "tick frequency: " << dFrq << std::endl;
	std::cout << "time(s): " << i64TickCount / dFrq << std::endl;

	//cv::B ^= cv::Scalar::all(255);
	cv::imshow("Image", A);
	cv::imshow("Correlation", corr);

	cv::waitKey();
	return 0;
}
*/
/*
// Exercise 12-8
// 参考： samples/cpp/grabcut.cpp
using namespace std;
using namespace cv;

static void help()
{
	cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
		"and then grabcut will attempt to segment it out.\n"
		"Call:\n"
		"./grabcut <image_name>\n"
		"\nSelect a rectangular area around the object you want to segment\n" <<
		"\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tn - next iteration\n"
		"\n"
		"\tleft mouse button - set rectangle\n"
		"\n"
		"\tCTRL+left mouse button - set GC_BGD pixels\n"
		"\tSHIFT+left mouse button - set GC_FGD pixels\n"
		"\n"
		"\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
		"\tSHIFT+right mouse button - set GC_PR_FGD pixels\n" << endl;
}

const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN = Scalar(0, 255, 0);

const int BGD_KEY = EVENT_FLAG_CTRLKEY;
const int FGD_KEY = EVENT_FLAG_SHIFTKEY;

static void getBinMask(const Mat& comMask, Mat& binMask)
{
	if (comMask.empty() || comMask.type() != CV_8UC1)
		CV_Error(Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
		binMask.create(comMask.size(), CV_8UC1);
	binMask = comMask & 1;
}

class GCApplication
{
public:
	enum { NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
	static const int radius = 2;
	static const int thickness = -1;

	void reset();
	void setImageAndWinName(const Mat& _image, const string& _winName);
	void showImage() const;
	void mouseClick(int event, int x, int y, int flags, void* param);
	int nextIter();
	int getIterCount() const { return iterCount; }
private:
	void setRectInMask();
	void setLblsInMask(int flags, Point p, bool isPr);

	const string* winName;
	const Mat* image;
	Mat mask;
	Mat bgdModel, fgdModel;

	uchar rectState, lblsState, prLblsState;
	bool isInitialized;

	Rect rect;
	vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
	int iterCount;
};

void GCApplication::reset()
{
	if (!mask.empty())
		mask.setTo(Scalar::all(GC_BGD));
	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear();  prFgdPxls.clear();

	isInitialized = false;
	rectState = NOT_SET;
	lblsState = NOT_SET;
	prLblsState = NOT_SET;
	iterCount = 0;
}

void GCApplication::setImageAndWinName(const Mat& _image, const string& _winName)
{
	if (_image.empty() || _winName.empty())
		return;
	image = &_image;
	winName = &_winName;
	mask.create(image->size(), CV_8UC1);
	reset();
}

void GCApplication::showImage() const
{
	if (image->empty() || winName->empty())
		return;

	Mat res;
	Mat binMask;
	if (!isInitialized)
		image->copyTo(res);
	else
	{
		getBinMask(mask, binMask);
		image->copyTo(res, binMask);
	}

	vector<Point>::const_iterator it;
	for (it = bgdPxls.begin(); it != bgdPxls.end(); ++it)
		circle(res, *it, radius, BLUE, thickness);
	for (it = fgdPxls.begin(); it != fgdPxls.end(); ++it)
		circle(res, *it, radius, RED, thickness);
	for (it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it)
		circle(res, *it, radius, LIGHTBLUE, thickness);
	for (it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it)
		circle(res, *it, radius, PINK, thickness);

	if (rectState == IN_PROCESS || rectState == SET)
		rectangle(res, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), GREEN, 2);

	imshow(*winName, res);
}

void GCApplication::setRectInMask()
{
	CV_Assert(!mask.empty());
	mask.setTo(GC_BGD);
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols - rect.x);
	rect.height = min(rect.height, image->rows - rect.y);
	(mask(rect)).setTo(Scalar(GC_PR_FGD));
}

void GCApplication::setLblsInMask(int flags, Point p, bool isPr)
{
	vector<Point>* bpxls, * fpxls;
	uchar bvalue, fvalue;
	if (!isPr)
	{
		bpxls = &bgdPxls;
		fpxls = &fgdPxls;
		bvalue = GC_BGD;
		fvalue = GC_FGD;
	}
	else
	{
		bpxls = &prBgdPxls;
		fpxls = &prFgdPxls;
		bvalue = GC_PR_BGD;
		fvalue = GC_PR_FGD;
	}
	if (flags & BGD_KEY)
	{
		bpxls->push_back(p);
		circle(mask, p, radius, bvalue, thickness);
	}
	if (flags & FGD_KEY)
	{
		fpxls->push_back(p);
		circle(mask, p, radius, fvalue, thickness);
	}
}

void GCApplication::mouseClick(int event, int x, int y, int flags, void*)
{
	// TODO add bad args check
	switch (event)
	{
	case EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if (rectState == NOT_SET && !isb && !isf)
		{
			rectState = IN_PROCESS;
			rect = Rect(x, y, 1, 1);
		}
		if ((isb || isf) && rectState == SET)
			lblsState = IN_PROCESS;
	}
	break;
	case EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if ((isb || isf) && rectState == SET)
			prLblsState = IN_PROCESS;
	}
	break;
	case EVENT_LBUTTONUP:
		if (rectState == IN_PROCESS)
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			rectState = SET;
			setRectInMask();
			CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();
		}
		if (lblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), false);
			lblsState = SET;
			showImage();
		}
		break;
	case EVENT_RBUTTONUP:
		if (prLblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), true);
			prLblsState = SET;
			showImage();
		}
		break;
	case EVENT_MOUSEMOVE:
		if (rectState == IN_PROCESS)
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();
		}
		else if (lblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), false);
			showImage();
		}
		else if (prLblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), true);
			showImage();
		}
		break;
	}
}

int GCApplication::nextIter()
{
	if (isInitialized)
		grabCut(*image, mask, rect, bgdModel, fgdModel, 1);
	else
	{
		if (rectState != SET)
			return iterCount;

		if (lblsState == SET || prLblsState == SET)
			grabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK);
		else
			grabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);

		isInitialized = true;
	}
	iterCount++;

	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear(); prFgdPxls.clear();

	return iterCount;
}

GCApplication gcapp;

static void on_mouse(int event, int x, int y, int flags, void* param)
{
	gcapp.mouseClick(event, x, y, flags, param);
}

int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, "{@input| messi5.jpg |}");
	help();

	string filename = parser.get<string>("@input");
	if (filename.empty())
	{
		cout << "\nDurn, empty filename" << endl;
		return 1;
	}
	Mat image = imread(samples::findFile(filename), IMREAD_COLOR);
	if (image.empty())
	{
		cout << "\n Durn, couldn't read image filename " << filename << endl;
		return 1;
	}

	const string winName = "image";
	namedWindow(winName, WINDOW_AUTOSIZE);
	setMouseCallback(winName, on_mouse, 0);

	gcapp.setImageAndWinName(image, winName);
	gcapp.showImage();

	for (;;)
	{
		char c = (char)waitKey(0);
		switch (c)
		{
		case '\x1b':
			cout << "Exiting ..." << endl;
			goto exit_main;
		case 'r':
			cout << endl;
			gcapp.reset();
			gcapp.showImage();
			break;
		case 'n':
			int iterCount = gcapp.getIterCount();
			cout << "<" << iterCount << "... ";
			int newIterCount = gcapp.nextIter();
			if (newIterCount > iterCount)
			{
				gcapp.showImage();
				cout << iterCount << ">" << endl;
			}
			else
				cout << "rect must be determined>" << endl;
			break;
		}
	}

exit_main:
	destroyWindow(winName);
	return 0;
}
*/
// Exercise 12-10
int main()
{
	cv::Size size49(49, 49);
	cv::Mat src, gray, dst;
	src = cv::imread("fruits.jpg");

	cv::pyrMeanShiftFiltering(src, dst, 20, 40, 2);

	cv::imshow("src", src);
	cv::imshow("dst", dst);

	cv::waitKey();

	return 0;
}
/*
// Exercise 12-11
// todo...
int main()
{
	cv::Scalar WHITE(255, 255, 255);
	cv::Point pt1, pt2, center;
	cv::Mat src, dst, rot_mat, src_r, labels;
	
	center.x = 100;
	center.y = 100;
	rot_mat = cv::getRotationMatrix2D(center, 0, 1);
	
	src = cv::Mat::zeros(200, 200, CV_8UC1);

	pt1.x = 10;
	pt1.y = 10;
	pt2.x = 80;
	pt2.y = 80;
	cv::rectangle(src, pt1, pt2, WHITE, cv::FILLED);
	cv::warpAffine(src, src_r, rot_mat, src.size());

	cv::distanceTransform(src_r, dst, labels, cv::DistanceTypes::DIST_L2, 5);

	std::cout << "before normalize:\n" << dst.rowRange(100, 120).colRange(100, 110) << std::endl;
	cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
	std::cout << "after normalize:\n" << dst.rowRange(100, 120).colRange(100, 110) << std::endl;

	cv::imshow("src", src);
	cv::imshow("src_r", src_r);
	cv::imshow("dst", dst);
	//cv::imshow("labels", labels);

	cv::waitKey();

	return 0;
}
*/
// Exercise 12-12
// lighting