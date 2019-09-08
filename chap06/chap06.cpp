// chap06.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
/*
int main() 
{ 
	cv::Mat src = cv::Mat::zeros(500, 500, CV_8UC3);
	cv::Point root_points[1][6];
	root_points[0][0] = cv::Point(215, 220);
	root_points[0][1] = cv::Point(460, 225);
	root_points[0][2] = cv::Point(466, 450);
	root_points[0][3] = cv::Point(235, 465);
	root_points[0][4] = cv::Point(260, 390);
	root_points[0][5] = cv::Point(96, 310);
	const cv::Point* ppt[1] = { root_points[0] };
	int npt[] = { 6 };
	polylines(src, ppt, npt, 1, 1, cv::Scalar(0, 0, 0), 1, 8, 0);
	imshow("Test", src);
	cv::Mat mask_ann, dst;
	src.copyTo(mask_ann);
	mask_ann.setTo(cv::Scalar::all(0));
	fillPoly(mask_ann, ppt, npt, 1, cv::Scalar(255, 255, 255));
	imshow("mask_ann", mask_ann);
	src.copyTo(dst, mask_ann);
	imshow("dst", dst);
	cv::waitKey();

	return 0; 
}
*/
/*
// Exercise 6-1
int main()
{
	int c = 0;

	uchar ucR = 0;
	uchar ucG = 0;
	uchar ucB = 0;

	int shape = 0;
	int thickness = 1;
	int lineType = cv::LineTypes::LINE_8;

	srand(time(0));

	cv::Mat img = cv::Mat::zeros(800, 800, CV_8UC3);
	cv::Point center(250, 250);

	cv::Point rootPoint[1][5];
	rootPoint[0][0] = cv::Point(10, 10);
	rootPoint[0][1] = cv::Point(400, 10);
	rootPoint[0][2] = cv::Point(400, 100);
	rootPoint[0][3] = cv::Point(200, 150);
	rootPoint[0][4] = cv::Point(100, 100);
	const cv::Point* ppt[1] = { rootPoint[0] };
	int npt[] = { 5 };

	std::vector<cv::Point> points;
	points.push_back(cv::Point(10, 10));
	points.push_back(cv::Point(400, 10));
	points.push_back(cv::Point(400, 100));
	points.push_back(cv::Point(200, 50));
	points.push_back(cv::Point(100, 100));

	std::vector<std::vector<cv::Point>> pvec;
	pvec.push_back(points);

	int arri[] = { 5 };
	cv::Point arrPoints[1][5] = {
		cv::Point(10, 10), cv::Point(400, 10), cv::Point(400, 100),
		cv::Point(200, 150), cv::Point(100, 100)
	};
	const cv::Point* arrpp[1] = { arrPoints[0] };
	//arrpp[0] = { arrPoints[0] };

	while (true) {
		ucR = rand() % 255;
		ucG = rand() % 255;
		ucB = rand() % 255;

		cv::Scalar color(ucB, ucG, ucR, 0);

		thickness = rand() % 8 + 1;
		std::cout << "thickness: " << thickness << std::endl;
		lineType = rand() % 4;
		switch (lineType) {
		case 0:
			lineType = cv::LineTypes::FILLED;
			std::cout << "LineTypes: FILLED" << std::endl;
			break;
		case 1:
			lineType = cv::LineTypes::LINE_4;
			std::cout << "LineTypes: LINE_4" << std::endl;
			break;
		case 2:
			lineType = cv::LineTypes::LINE_8;
			std::cout << "LineTypes: LINE_8" << std::endl;
			break;
		case 3:
			lineType = cv::LineTypes::LINE_AA;
			std::cout << "LineTypes: LINE_AA" << std::endl;
			break;
		default:
			lineType = cv::LineTypes::LINE_8;
			std::cout << "LineTypes: LINE_8" << std::endl;
			break;
		}

		//shape = rand() % 7;
		shape = 6;

		switch (shape) {
		case 0:
			cv::circle(img, center, 200, color, thickness, lineType);
			break;

		case 1:
			cv::ellipse(img, center, cv::Size(250, 150), 30, 0, 360, color, thickness, lineType);
			break;

		case 2:
			// fillPoly 和fillConvexPoly 的LineType 只能用4 或8
			if (lineType == cv::LineTypes::FILLED) {
				lineType = cv::LineTypes::LINE_4;
			}
			else if (lineType == cv::LineTypes::LINE_AA) {
				lineType = cv::LineTypes::LINE_8;
			}
			std::cout << "shape 2, lineType " << lineType << std::endl;
			cv::fillConvexPoly(img, points, color, lineType);
			break;

		case 3:
			// fillPoly 和fillConvexPoly 的LineType 只能用4 或8
			if (lineType == cv::LineTypes::FILLED) {
				lineType = cv::LineTypes::LINE_4;
			}
			else if (lineType == cv::LineTypes::LINE_AA) {
				lineType = cv::LineTypes::LINE_8;
			}
			std::cout << "shape 3, lineType " << lineType << std::endl;
			// 注意pvec 不是InputArray，而是InputArrayOfArrays
			cv::fillPoly(img, pvec, color, lineType);
			//cv::fillPoly(img, (const cv::Point**)&arrpp[0], &arri[0], 1, color, lineType);
			//cv::fillPoly(img, arrpp, arri, 1, color, lineType);
			//cv::fillPoly(img, ppt, npt, 1, color, lineType);
			// 上面这几个注释掉的使用任何一个都可以正常运行
			break;

		case 4:
			if (lineType == cv::LineTypes::FILLED) {
				lineType = cv::LineTypes::LINE_4;
			}
			std::cout << "shape 4, lineType " << lineType << std::endl;
			cv::line(img, points[0], points[1], color, thickness, lineType);
			break;

		case 5:
			if (lineType == cv::LineTypes::FILLED) {
				lineType = cv::LineTypes::LINE_4;
			}
			std::cout << "shape 5, lineType " << lineType << std::endl;
			cv::rectangle(img, points[0], points[2], color, thickness, lineType);
			break;

		case 6:
			//cv::polylines(img, points, false, color, thickness, lineType);
			cv::polylines(img, points, true, color, thickness, lineType);
			break;

		default:
			cv::Scalar color(ucB, ucG, ucR, 0);
			cv::circle(img, center, 200, color, thickness, lineType);
			break;
		}
		
		cv::namedWindow("Exercise6-1", 1);
		cv::imshow("Exercise6-1", img);

		c = cv::waitKey(0);

		if (27 == c) {
			break;
		}

		// reset
		//img.setTo(0);
		img = cv::Mat::zeros(800, 800, CV_8UC3);
	}

	return 0;
}
*/
/*
// Exercise 6-2
int main(int argc, char** argv) {
	//std::cout << argv[1] << std::endl;

	//cv::Mat img = cv::imread(argv[1], -1);
	//cv::Mat img = cv::imread("D:\\work\\SV\\1.bmp", -1);
	cv::Mat img(4, 5, CV_8UC3, cv::Scalar(255, 0, 0));
	cv::Mat dst(img);

	if (img.empty()) {
		return -1;
	}

	cv::namedWindow("Exercise6-2", cv::WINDOW_AUTOSIZE);
	cv::imshow("Exercise6-2", img);

	cv::cvtColor(img, dst, cv::COLOR_BGR2GRAY);
	// 此时dst 是单通道图像

	cv::cvtColor(dst, dst, cv::COLOR_GRAY2RGB);
	// 此时dst 是三通道图像，但看起来是灰色的

	srand(time(0));
	cv::putText(dst, "this is a gray bmp.", cv::Point(100, 100), 
		cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));
		//cv::Scalar(rand() % 255, rand() % 255, rand() % 255));

	cv::namedWindow("Exercise6-2-gray", cv::WINDOW_AUTOSIZE);
	cv::imshow("Exercise6-2-gray", dst);

	cv::waitKey(0);
	cv::destroyWindow("Exercise6-2");
	cv::destroyWindow("Exercise6-2-gray");

	return 0;
}
*/
/*
// Exercise 6-3
// 动态调整fps，吼吼。。
int main(int argc, char** argv) {
	//std::cout << argv[1] << std::endl;

	cv::namedWindow("Exercise6-3", cv::WINDOW_AUTOSIZE);
	cv::VideoCapture cap;
	//cap.open(cv::String(argv[1]));
	cap.open("D:\\work\\SV\\SampleVideo_1280x720_30mb.mp4");
	
	double fps_stand = cap.get(cv::CAP_PROP_FPS);
	double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "fps: " << fps_stand << " w: " << width 
		<< " h: " << height<< std::endl;

	cv::Mat frame;

	int64 i64Count = 0;
	int64 i64Cur = 0;
	int64 i64Begin = cv::getTickCount();
	cv::Point ori(100, 100);
	cv::Scalar color(255, 0, 0);
	cv::String text;
	char buf[256] = { 0 };
	double dFrq = cv::getTickFrequency();
	double fps = 0;
	int delay = 33;
	for (;;) {
		cap >> frame;
		if (frame.empty()) {
			break;
		}

		i64Cur = cv::getTickCount();
		if (i64Count > 0) {
			//dFrq = cv::getTickFrequency();
			//std::cout << "dFrq: " << dFrq << std::endl;
			fps = (double)i64Count / ((i64Cur - i64Begin)/dFrq);
			if (fps < fps_stand) {
				delay--;
			}
			else if (fps > fps_stand) {
				delay++;
			}
			if (0 == delay) {
				delay++;
			}
			std::cout << "delay: " << delay << std::endl;
			srand(time(0));
			
			sprintf_s(buf, "fps: %f", fps);
			text.clear();
			text.append(buf);

			cv::putText(frame, text, ori,
				cv::FONT_HERSHEY_SIMPLEX, 1, color);
		}
		i64Count++;

		cv::imshow("Exercise6-3", frame);

		if (cv::waitKey(delay) >= 0) {
			break;
		}
	}

	return 0;
}
*/
/*
// Exercise 6-4
// 修改自Exercise 6-1
int main()
{
	int c = 0;

	uchar ucR = 0;
	uchar ucG = 0;
	uchar ucB = 0;

	int shape = 0;
	int thickness = 1;
	int lineType = cv::LineTypes::LINE_8;

	srand(time(0));

	cv::Mat img = cv::Mat::zeros(800, 800, CV_8UC3);
	cv::Point center(250, 250);

	cv::Point rootPoint[1][5];
	rootPoint[0][0] = cv::Point(10, 10);
	rootPoint[0][1] = cv::Point(400, 10);
	rootPoint[0][2] = cv::Point(400, 100);
	rootPoint[0][3] = cv::Point(200, 150);
	rootPoint[0][4] = cv::Point(100, 100);
	const cv::Point* ppt[1] = { rootPoint[0] };
	int npt[] = { 5 };

	std::vector<cv::Point> points;
	points.push_back(cv::Point(10, 10));
	points.push_back(cv::Point(400, 10));
	points.push_back(cv::Point(400, 100));
	points.push_back(cv::Point(200, 50));
	points.push_back(cv::Point(100, 100));

	std::vector<std::vector<cv::Point>> pvec;
	pvec.push_back(points);

	int arri[] = { 5 };
	cv::Point arrPoints[1][5] = {
		cv::Point(10, 10), cv::Point(400, 10), cv::Point(400, 100),
		cv::Point(200, 150), cv::Point(100, 100)
	};
	const cv::Point* arrpp[1] = { arrPoints[0] };
	//arrpp[0] = { arrPoints[0] };

	lineType = rand() % 4;
	shape = rand() % 7;

	while (true) {
		ucR = rand() % 255;
		ucG = rand() % 255;
		ucB = rand() % 255;

		cv::Scalar color(ucB, ucG, ucR, 0);

		thickness = rand() % 8 + 1;
		std::cout << "thickness: " << thickness << std::endl;
		
		switch (lineType) {
		case 0:
			lineType = cv::LineTypes::FILLED;
			std::cout << "LineTypes: FILLED" << std::endl;
			break;
		case 1:
			lineType = cv::LineTypes::LINE_4;
			std::cout << "LineTypes: LINE_4" << std::endl;
			break;
		case 2:
			lineType = cv::LineTypes::LINE_8;
			std::cout << "LineTypes: LINE_8" << std::endl;
			break;
		case 3:
			lineType = cv::LineTypes::LINE_AA;
			std::cout << "LineTypes: LINE_AA" << std::endl;
			break;
		default:
			lineType = cv::LineTypes::LINE_8;
			std::cout << "LineTypes: LINE_8" << std::endl;
			break;
		}

		switch (shape) {
		case 0:
			cv::circle(img, center, 200, color, thickness, lineType);
			break;

		case 1:
			cv::ellipse(img, center, cv::Size(250, 150), 30, 0, 360, color, thickness, lineType);
			break;

		case 2:
			// fillPoly 和fillConvexPoly 的LineType 只能用4 或8
			if (lineType == cv::LineTypes::FILLED) {
				lineType = cv::LineTypes::LINE_4;
			}
			else if (lineType == cv::LineTypes::LINE_AA) {
				lineType = cv::LineTypes::LINE_8;
			}
			std::cout << "shape 2, lineType " << lineType << std::endl;
			cv::fillConvexPoly(img, points, color, lineType);
			break;

		case 3:
			// fillPoly 和fillConvexPoly 的LineType 只能用4 或8
			if (lineType == cv::LineTypes::FILLED) {
				lineType = cv::LineTypes::LINE_4;
			}
			else if (lineType == cv::LineTypes::LINE_AA) {
				lineType = cv::LineTypes::LINE_8;
			}
			std::cout << "shape 3, lineType " << lineType << std::endl;
			// 注意pvec 不是InputArray，而是InputArrayOfArrays
			cv::fillPoly(img, pvec, color, lineType);
			//cv::fillPoly(img, (const cv::Point**)&arrpp[0], &arri[0], 1, color, lineType);
			//cv::fillPoly(img, arrpp, arri, 1, color, lineType);
			//cv::fillPoly(img, ppt, npt, 1, color, lineType);
			// 上面这几个注释掉的使用任何一个都可以正常运行
			break;

		case 4:
			if (lineType == cv::LineTypes::FILLED) {
				lineType = cv::LineTypes::LINE_4;
			}
			std::cout << "shape 4, lineType " << lineType << std::endl;
			cv::line(img, points[0], points[1], color, thickness, lineType);
			break;

		case 5:
			if (lineType == cv::LineTypes::FILLED) {
				lineType = cv::LineTypes::LINE_4;
			}
			std::cout << "shape 5, lineType " << lineType << std::endl;
			cv::rectangle(img, points[0], points[2], color, thickness, lineType);
			break;

		case 6:
			if (lineType == cv::LineTypes::FILLED) {
				lineType = cv::LineTypes::LINE_4;
			}
			//cv::polylines(img, points, false, color, thickness, lineType);
			cv::polylines(img, points, true, color, thickness, lineType);
			break;

		default:
			cv::Scalar color(ucB, ucG, ucR, 0);
			cv::circle(img, center, 200, color, thickness, lineType);
			break;
		}

		cv::namedWindow("Exercise6-1", 1);
		cv::imshow("Exercise6-1", img);

		c = cv::waitKey(0);

		if (27 == c) {
			break;
		}

		if ('c' == c || 'C' == c) {
			shape = 0;
		}
		else if ('e' == c || 'E' == c) {
			shape = 1;
		}
		else if ('p' == c || 'P' == c) {
			shape = 2;
		}
		else if ('o' == c || 'O' == c) {
			shape = 3;
		}
		else if ('l' == c || 'L' == c) {
			shape = 4;
		}
		else if ('r' == c || 'R' == c) {
			shape = 5;
		}
		else if ('y' == c || 'Y' == c) {
			shape = 6;
		}
		else if ('1' == c) {
			lineType = 0;
		}
		else if ('2' == c) {
			lineType = 1;
		}
		else if ('3' == c) {
			lineType = 2;
		}
		else if ('4' == c) {
			lineType = 3;
		}

		// reset
		//img.setTo(0);
		img = cv::Mat::zeros(800, 800, CV_8UC3);
	}

	return 0;
}
*/
// Exercise 6-5
int main()
{
	size_t sizePoints = 0;
	cv::Size axes(200, 200);
	cv::Scalar white(255, 255, 255);
	std::vector<cv::Point> vecPoints;
	cv::Mat img = cv::Mat::zeros(500, 500, CV_8UC3);

	cv::Point ptCenter(250, 250);
	//cv::Point pt2(450, 250);
	//cv::LineIterator li(img, ptCenter, pt2);
	//std::cout << "li.count: " << li.count << std::endl;

	cv::ellipse2Poly(ptCenter, axes, 0, 0, 360, 1, vecPoints);
	sizePoints = vecPoints.size();
	std::cout << "points.count: " << sizePoints << std::endl;

	int no = 0;
	for (std::vector<cv::Point>::const_iterator ci = vecPoints.begin();
		ci != vecPoints.end(); ci++, no++) {
		img.at<cv::Vec3b>(ci->x, ci->y) = 0xffffff;
		//img.at<cv::Vec3b>(0, 0) = 0xffffff;
		//img.at<cv::Vec3b>(0, 0)[1] = 0xff;
		//img.at<cv::Vec3b>(0, 0)[2] = 0xff;
		//img.at<cv::Vec3b>(0, 1) = 0xffffff;
		cv::LineIterator li4(img, ptCenter, *ci, 4);
		std::cout << "no" << no << " li4.count: " << li4.count << std::endl;

		cv::LineIterator li8(img, ptCenter, *ci, 8);
		std::cout << "no" << no << " li8.count: " << li8.count << std::endl;

		if (li4.count > li8.count) {
			std::cout << "44444444444444444444 is more." << std::endl;
		}
		else if (li4.count < li8.count) {
			std::cout << "88888888888888888888 is more." << std::endl;
		}
		else if (li4.count == li8.count) {
			std::cout << "****************************** no" << no << " li4.count == li8.count " << std::endl;
		}
	}

	cv::namedWindow("Exercise6-5", 1);
	cv::imshow("Exercise6-5", img);

	cv::waitKey();

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
