// chap11.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
/*
// Example 11-1
// 仿射变换
int main(int argc, char** argv)
{
	if (argc != 2) {
		std::cout << "Warp affine\nUsage: " << argv[0]
			<< " <imagename>\n" << std::endl;
		return -1;
	}

	cv::Mat src = cv::imread(argv[1], 1);
	if (src.empty()) {
		std::cout << "Can not load " << argv[1] << std::endl;
		return -1;
	}

	cv::Point2f srcTri[] = {
		cv::Point2f(0, 0),
		cv::Point2f(src.cols - 1, 0),
		cv::Point2f(0, src.rows - 1)
	};

	cv::Point2f dstTri[] = {
		cv::Point2f(src.cols * 0.f, src.rows * 0.33f),
		cv::Point2f(src.cols * 0.85f, src.rows * 0.25f),
		cv::Point2f(src.cols * 0.15f, src.rows * 0.7f)
	};

	cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
	cv::Mat dst, dst2;
	cv::warpAffine(src, dst, warp_mat, src.size(),
		cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

	for (int i = 0; i < 3; i++) {
		cv::circle(dst, dstTri[i], 5, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
	}

	cv::imshow("Affine Transform Test", dst);
	//cv::waitKey();

	for (int frame = 0; ; frame++) {
		cv::Point2f center(src.cols * 0.5f, src.rows * 0.5f);
		double angle = frame * 3 % 360;
		double scale = (cos((angle - 60) * 3.1415926 / 180) + 1.05) * 0.8;
		cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);

		cv::warpAffine(src, dst, rot_mat, src.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

		cv::imshow("Rotated Image", dst);
		if (cv::waitKey(30) >= 0) {
			break;
		}
	}

	return 0;
}
*/
/*
// Example 11-2
// 透视变换
int main(int argc, char **argv)
{
	if (argc != 2) {
		std::cout << "Perspective Warp\nUsage: " << argv[0] 
			<< " <imagename>\n" << std::endl;
		return -1;
	}

	cv::Mat src = cv::imread(argv[1], 1);
	if (src.empty()) {
		std::cout << "Can not load " << argv[1] << std::endl;
		return -1;
	}

	cv::Point2f srcQuad[] = {
		cv::Point2f(0, 0),
		cv::Point2f(src.cols - 1, 0),
		cv::Point2f(src.cols - 1, src.rows - 1),
		cv::Point2f(0, src.rows - 1)
	};

	cv::Point2f dstQuad[] = {
		cv::Point2f(src.cols*0.05f, src.rows*0.33f),
		cv::Point2f(src.cols * 0.9f, src.rows * 0.25f),
		cv::Point2f(src.cols * 0.8f, src.rows * 0.9f),
		cv::Point2f(src.cols * 0.2f, src.rows * 0.7f)
	};

	cv::Mat warp_mat = cv::getPerspectiveTransform(srcQuad, dstQuad);
	cv::Mat dst;
	cv::warpPerspective(src, dst, warp_mat, src.size(), 
		cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

	for (int i = 0; i < 4; i++) {
		cv::circle(dst, dstQuad[i], 5, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
	}

	cv::imshow("Perspective Transform Test", dst);
	cv::waitKey();

	return 0;
}
*/
/*
// Example 11-3
int main(int argc, char** argv)
{
	//if (argc != 3) {
		//std::cout << "LogPolar\nUsage: " << argv[0] << " <imagename> <M value>\n"
			//<< "<M value>~30 is usually good enough\n";
		//return -1;
	//}

	//cv::Mat src = cv::imread(argv[1], 1);
	cv::Mat src = cv::imread("1.jpg", 1);

	if (src.empty()) {
		std::cout << "Can not load " << argv[1] << std::endl;
		return -1;
	}

	//double M = atof(argv[2]);
	double M = 30;

	cv::Mat dst(src.size(), src.type()), src2(src.size(), src.type());

	cv::logPolar(src, dst, cv::Point2f(src.cols * 0.5f, src.rows * 0.5f),
		M, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);

	cv::logPolar(dst, src2, cv::Point2f(src.cols * 0.5f, src.rows * 0.5f), 
		M, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);

	cv::imshow("src", src);
	cv::imshow("log-polar", dst);
	cv::imshow("inverse log-polar", src2);

	cv::waitKey();
	return 0;
}
*/
/*
// Exercise 11-1
int main()
{
	int ddepth = CV_8U;
	cv::Mat raw, gray, invert, laplace;

	raw = cv::imread("face.jpg");
	cv::cvtColor(raw, gray, cv::COLOR_BGR2GRAY);
	cv::bitwise_not(gray, invert);

	cv::Laplacian(invert, laplace, ddepth);

	cv::imshow("raw", raw);
	cv::imshow("gray", gray);
	cv::imshow("invert", invert);
	cv::imshow("laplace", laplace);

	cv::waitKey();
	return 0;
}
*/
/*
// Exercise 11-2
// 1)
//              theta
//              .     .
//              .    .
//              |   /
//              I
//              I
//              .
//              .
//              I
//              I
//   -------    |    \
//   |     |    |     \
//   o_____|    |     /
//              |____/_____logr
// note: 中间一段I 代表值为0
//
// 2)
//              theta
//              .
//              .
//              |    \
//              |     \
//              |     /
//              |    /
//              |   /
//              |  |
//    ~   ~     |   \
//  ( o_____)   |    \
//    @    @    |_____\____logr
//
//
// 3)
//              theta
//              .
//              .
//              |    /
//              |   /
//              I
//              .
//              I
//    ~   ~     |   \
//o ( ______)   |    \
//    @    @    |_____\____logr
//
int main()
{
	int ddepth = CV_8U;
	cv::Mat rect, circle, rect_pl, circle_pl_inside, circle_pl_outside;

	cv::Rect r(50, 50, 100, 100);
	cv::Scalar white(255, 255, 255);
	cv::Point rect_lb(50, 150);
	cv::Point center(100, 100);
	cv::Point inside(50, 100);
	cv::Point outside(0, 100);

	rect = cv::Mat::zeros(200, 200, CV_8UC1);
	circle = cv::Mat::zeros(200, 200, CV_8UC1);
	rect_pl = cv::Mat::zeros(200, 200, CV_8UC1);
	circle_pl_inside = cv::Mat::zeros(200, 200, CV_8UC1);
	circle_pl_outside = cv::Mat::zeros(200, 200, CV_8UC1);

	cv::rectangle(rect, r, white);
	cv::circle(circle, center, 50, white);

	cv::logPolar(rect, rect_pl, rect_lb, 100, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);
	cv::logPolar(circle, circle_pl_inside, inside, 100, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);
	cv::logPolar(circle, circle_pl_outside, outside, 3, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);
	
	cv::imshow("rect", rect);
	cv::imshow("circle", circle);
	cv::imshow("rect_pl", rect_pl);
	cv::imshow("circle_pl_inside", circle_pl_inside);
	cv::imshow("circle_pl_outside", circle_pl_outside);

	cv::waitKey();
	return 0;
}
*/
// Exercise 11-3
// 参考：https://www.researchgate.net/publication/221416060_Fourier_signature_in_log-polar_images
/*
// Exercise 11-4
int main()
{
	int ddepth = CV_8U;
	cv::Mat rot_mat;
	cv::Mat big, big_r, small, small_r;
	cv::Mat big_lp, big_r_lp, small_lp, small_r_lp;

	cv::Rect rcBig(50, 50, 100, 100);
	cv::Rect rcSmall(75, 75, 50, 50);
	cv::Scalar white(255, 255, 255);
	cv::Point2f center(100., 100.);
	cv::Point2f center2(50., 100.);
	
	big = cv::Mat::zeros(200, 200, CV_8UC1);
	big_r = cv::Mat::zeros(200, 200, CV_8UC1);
	small = cv::Mat::zeros(200, 200, CV_8UC1);
	small_r = cv::Mat::zeros(200, 200, CV_8UC1);

	rot_mat = cv::getRotationMatrix2D(center, 45, 1);

	cv::rectangle(big, rcBig, white);
	cv::rectangle(small, rcSmall, white);

	cv::warpAffine(big, big_r, rot_mat, big.size());
	cv::warpAffine(small, small_r, rot_mat, small.size());

	//cv::logPolar(big, big_lp, center, 25, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);
	//cv::logPolar(big_r, big_r_lp, center, 25, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);
	cv::logPolar(big, big_lp, center2, 35, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);
	cv::logPolar(big_r, big_r_lp, center2, 35, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);
	cv::logPolar(small, small_lp, center, 25, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);
	cv::logPolar(small_r, small_r_lp, center, 25, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);
	//cv::logPolar(small, small_lp, center2, 25, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);
	//cv::logPolar(small_r, small_r_lp, center2, 25, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);

	cv::imshow("big", big);
	cv::imshow("big_r", big_r);
	cv::imshow("small", small);
	cv::imshow("small_r", small_r);

	cv::imshow("big_lp", big_lp);
	cv::imshow("big_r_lp", big_r_lp);
	cv::imshow("small_lp", small_lp);
	cv::imshow("small_r_lp", small_r_lp);

	cv::waitKey();
	return 0;
}
*/
/*
// Exercise 11-5
// 可以，下面代码中的dstQuad2 是逆时针旋转90度的点集，
// 对其使用透视变换后可得到旋转图像，
// 在dstQuad2 上同样可以施加透视效果，
// 所以可以同时完成
int main(int argc, char** argv)
{
	if (argc != 2) {
		std::cout << "Perspective Warp\nUsage: " << argv[0]
			<< " <imagename>\n" << std::endl;
		return -1;
	}

	cv::Mat src = cv::imread(argv[1], 1);
	if (src.empty()) {
		std::cout << "Can not load " << argv[1] << std::endl;
		return -1;
	}

	cv::Point2f srcQuad[] = {
		cv::Point2f(0, 0),
		cv::Point2f(src.cols - 1, 0),
		cv::Point2f(src.cols - 1, src.rows - 1),
		cv::Point2f(0, src.rows - 1)
	};

	cv::Point2f dstQuad[] = {
		cv::Point2f(src.cols * 0.05f, src.rows * 0.33f),
		cv::Point2f(src.cols * 0.9f, src.rows * 0.25f),
		cv::Point2f(src.cols * 0.8f, src.rows * 0.9f),
		cv::Point2f(src.cols * 0.2f, src.rows * 0.7f)
	};

	cv::Point2f dstQuad2[] = {
		cv::Point2f(src.rows-1, 0),
		cv::Point2f(src.rows-1, src.cols-1),
		cv::Point2f(0, src.cols-1),
		cv::Point2f(0, 0)
	};

	cv::Mat warp_mat = cv::getPerspectiveTransform(srcQuad, dstQuad);
	cv::Mat dst;
	cv::warpPerspective(src, dst, warp_mat, src.size(),
		cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

	for (int i = 0; i < 4; i++) {
		cv::circle(dst, dstQuad[i], 5, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
	}

	cv::Mat warp_mat2 = cv::getPerspectiveTransform(srcQuad, dstQuad2);
	cv::Mat dst2;
	cv::warpPerspective(src, dst2, warp_mat2, cv::Size(src.size().height, src.size().width),
		cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

	cv::imshow("src", src);
	cv::imshow("Perspective Transform Test", dst);
	cv::imshow("dst2", dst2);
	cv::waitKey();

	return 0;
}
*/
/*
// Exercise 11-6
int main()
{
	int ddepth = CV_8U;
	cv::Mat raw, gray, mask, dst;

	raw = cv::imread("11-6.jpg");
	cv::cvtColor(raw, gray, cv::COLOR_BGR2GRAY);

	cv::threshold(gray, mask, 235, 255, cv::THRESH_BINARY);

	cv::inpaint(raw, mask, dst, 6, cv::INPAINT_TELEA);
	
	cv::imshow("raw", raw);
	cv::imshow("gray", gray);
	cv::imshow("mask", mask);
	cv::imshow("dst", dst);

	cv::waitKey();
	return 0;
}
*/
// Exercise 11-7
int main()
{
	int ddepth = CV_8U;
	cv::Mat raw, gray, mask, dst;

	raw = cv::imread("11-7.jpg");
	cv::cvtColor(raw, gray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(gray, dst);
	cv::imshow("raw", raw);
	cv::imshow("gray", gray);
	cv::imshow("dst", dst);

	cv::waitKey();
	return 0;
}
// Exercise 11-8
// 直方图均衡可增强对比度，方法是展开整体的强度分布
// 去噪是基于像素周围的平均值，而不是整体展开什么