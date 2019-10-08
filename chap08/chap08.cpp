// chap08.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
/*
// example 8-2
int main()
{
	cv::FileStorage fs("test.yml", cv::FileStorage::WRITE);

	fs << "frameCount" << 5;

	char bufTime[32] = { 0 };
	tm tmRaw;
	time_t rawtime;
	time(&rawtime);
	localtime_s(&tmRaw, &rawtime);
	asctime_s(bufTime, 32, &tmRaw);
	fs << "calibrationDate" << bufTime;

	cv::Mat cameraMatrix = (
		cv::Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1
	);

	cv::Mat distCoeffs = (
		cv::Mat_<double>(5, 1) << 0.1, 0.01, -0.001, 0, 0
	);

	fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;

	fs << "features" << "[";
	for (int i = 0; i < 3; i++) {
		int x = rand() % 640;
		int y = rand() % 320;
		uchar lbp = rand() % 256;

		fs << "{:" << "x" << x << "y" << y << "lbp" << "[:";
		for (int j = 0; j < 8; j++) {
			fs << ((lbp >> j) & 1);
		}
		fs << "]" << "}";
	}
	fs << "]";
	fs.release();

	return 0;
}
*/
/*
// example 8-2
int main()
{
	cv::FileStorage fs("test.yml", cv::FileStorage::READ);

	int frameCount = (int)fs["frameCount"];

	std::string date;
	fs["calibrationDate"] >> date;

	cv::Mat cameraMatrix, distCoeffs;
	fs["cameraMatrix"] >> cameraMatrix;
	fs["distCoeffs"] >> distCoeffs;

	std::cout << "frameCount: " << frameCount << std::endl;
	std::cout << "calibration date: " << date << std::endl;
	std::cout << "camera matrix: " << cameraMatrix << std::endl;
	std::cout << "distortion coeffs: " << distCoeffs << std::endl;

	cv::FileNode features = fs["features"];
	cv::FileNodeIterator it = features.begin(), it_end = features.end();
	std::vector<uchar> lbpval;
	int idx = 0;

	for (; it != it_end; it++, idx++) {
		std::cout << "feature #" << idx << ": ";
		std::cout << "x=" << (int)(*it)["x"] 
			<< ", y=" << (int)(*it)["y"] << ", lbp: (";
		(*it)["lbp"] >> lbpval;
		for (int i = 0; i < (int)lbpval.size(); i++) {
			std::cout << " " << (int)lbpval[i];
		}
		std::cout << ")" << std::endl;
	}

	fs.release();

	return 0;
}
*/
/*
// exercise 8-1
// 参考exercise 6-3 
int main(int argc, char** argv) {
	//std::cout << argv[1] << std::endl;

	cv::namedWindow("Exercise8-1-(1)", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Exercise8-1-(2)", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Exercise8-1-(3)", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Exercise8-1-3In1", cv::WINDOW_AUTOSIZE);
	cv::VideoCapture cap;
	//cap.open(cv::String(argv[1]));
	//cap.open("D:\\work\\SV\\SampleVideo_1280x720_30mb.mp4");
	//cap.open("D:\\work\\SV\\big_buck_bunny_144p_10mb.3gp");
	cap.open("D:\\work\\SV\\big_buck_bunny_240p_30mb.mp4");

	double fps_stand = cap.get(cv::CAP_PROP_FPS);
	double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "fps: " << fps_stand << " w: " << width
		<< " h: " << height << std::endl;

	cv::Mat frame;
	cv::Mat gray;
	cv::Mat canny;
	cv::Mat threeInOne = cv::Mat::zeros(height, width * 3, CV_8UC3);
	cv::Mat left = threeInOne.colRange(0, width);
	cv::Mat middle = threeInOne.colRange(width, width*2);
	cv::Mat right = threeInOne.colRange(width*2, width*3);
	
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

		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		cv::cvtColor(gray, gray, cv::COLOR_GRAY2RGB);
		cv::Canny(frame, canny, 10, 100, 3, true);
		cv::cvtColor(canny, canny, cv::COLOR_GRAY2RGB);

		//frame.copyTo(threeInOne.colRange(0, width));
		//gray.copyTo(threeInOne.colRange(width, width*2));
		//canny.copyTo(threeInOne.colRange(width*2, width*3));
		frame.copyTo(left);
		gray.copyTo(middle);
		canny.copyTo(right);

		cv::putText(threeInOne, "raw", cv::Point(0, 20),
			cv::FONT_HERSHEY_SIMPLEX, 1, color);
		cv::putText(threeInOne, "gray", cv::Point(width, 20),
			cv::FONT_HERSHEY_SIMPLEX, 1, color);
		cv::putText(threeInOne, "canny", cv::Point(width * 2, 20),
			cv::FONT_HERSHEY_SIMPLEX, 1, color);

		i64Cur = cv::getTickCount();
		if (i64Count > 0) {
			//dFrq = cv::getTickFrequency();
			//std::cout << "dFrq: " << dFrq << std::endl;
			fps = (double)i64Count / ((i64Cur - i64Begin) / dFrq);
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

		cv::imshow("Exercise8-1-(1)", frame);
		cv::imshow("Exercise8-1-(2)", gray);
		cv::imshow("Exercise8-1-(3)", canny);
		cv::imshow("Exercise8-1-3In1", threeInOne);

		if (cv::waitKey(delay) >= 0) {
			break;
		}
	}

	return 0;
}
*/
/*
// exercise 8-2
// 参考博客：https://blog.csdn.net/jiamuju84/article/details/52893239
void on_mouse(int EVENT, int x, int y, int flags, void* userdata);
int main(int argc, char** argv) {
	cv::namedWindow("exercise8-2");
	cv::namedWindow("dashboard");
	cv::Mat src, frontend, dashboard;
	cv::Mat* params[3];

	dashboard = cv::Mat::zeros(100, 300, CV_8UC3);

	params[0] = &src;
	params[1] = &frontend;
	params[2] = &dashboard;

	src = cv::imread("D:\\work\\SV\\1.bmp");
	src.copyTo(frontend);
	cv::setMouseCallback("exercise8-2", on_mouse, &params[0]);
	//cv::setMouseCallback("exercise8-2", on_mouse, &src);

	while (true) {
		cv::imshow("exercise8-2", frontend);
		cv::imshow("dashboard", dashboard);
		cv::waitKey(40);
	}

	return 0;
}

void on_mouse(int EVENT, int x, int y, int flags, void* userdata)
{
	cv::Scalar color(0, 0, 255);
	cv::String text;
	cv::Mat hh;
	cv::Mat frontend;
	cv::Mat dashboard;

	char buf[64] = { 0 };

	hh = *(*(cv::Mat**)userdata);
	frontend = *(*((cv::Mat**)userdata+1));
	dashboard = *(*((cv::Mat * *)userdata + 2));
	cv::Point p(x, y);
	cv::Point ori;

	switch (EVENT) {
	case cv::EVENT_LBUTTONDOWN:
		printf("b=%d\t", hh.at<cv::Vec3b>(p)[0]);
		printf("g=%d\t", hh.at<cv::Vec3b>(p)[1]);
		printf("r=%d\n", hh.at<cv::Vec3b>(p)[2]);
		cv::circle(hh, p, 2, cv::Scalar(255), 3);
		break;

	case cv::EVENT_MOUSEMOVE:
		// 在当前鼠标位置右下/左上方(20,20) 处显示当前点颜色和坐标
		sprintf_s(buf, "b=%d g=%d r=%d x=%d y=%d", hh.at<cv::Vec3b>(p)[0], 
			hh.at<cv::Vec3b>(p)[1], hh.at<cv::Vec3b>(p)[2], x, y);
		text.clear();
		text.append(buf);
		//ori.x = x + 20;
		//ori.y = y + 20;
		hh.copyTo(frontend);
		//cv::putText(frontend, text, ori,
			//cv::FONT_HERSHEY_SIMPLEX, 1, color);
		dashboard = cv::Mat::zeros(100, 300, CV_8UC3);
		// 如果不想这样重新初始化
		// chap04.cpp 中有一个写好的清除背景的函数可以使用
		cv::putText(dashboard, text, cv::Point(0, 30),
			cv::FONT_HERSHEY_SIMPLEX, 1, color);
		break;
	}
}
*/
/*
// exercise 8-3
// 参考example 9-2
void my_mouse_callback(int event, int x, int y, int flags, void* param);

cv::Rect box;
bool drawing_box = false;

void draw_box(cv::Mat& img, cv::Rect& box)
{
	cv::rectangle(img, box.tl(), box.br(), cv::Scalar(0, 0, 255));
}

void help()
{
	std::cout << "Call: ./ch4_ext_1\n" <<
		" shows how to use a mouse to draw regions in an image." << std::endl;
}

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

			double gama_r = 1 - (alpha_r - alpha_min) / (1 - 3 * alpha_min);
			double gama_g = 1 - (alpha_g - alpha_min) / (1 - 3 * alpha_min);
			double gama_b = 1 - (alpha_b - alpha_min) / (1 - 3 * alpha_min);
			double gama =
				cv::max<double>(cv::max<double>(gama_r, gama_g), gama_b);

			double temp = (gama * (R + G + B) - maxC) / (3 * gama - 1);
			if (i < 10) {
				std::cout << " temp: " << temp;// << std::endl;
			}
			dst.at<cv::Vec3b>(i, j)[0] = B - (temp + 0.5);
			dst.at<cv::Vec3b>(i, j)[1] = G - (temp + 0.5);
			dst.at<cv::Vec3b>(i, j)[2] = R - (temp + 0.5);
		}
	}
}
*/
/**
 * 统计选中区域颜色信息
 * image [INPUT]
 * data  [OUTPUT]
 */
void statisticsColor(cv::Mat &image, cv::Mat &data)
{
	data.setTo(0);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			uchar B = image.at<cv::Vec3b>(i, j)[0];
			uchar G = image.at<cv::Vec3b>(i, j)[1];
			uchar R = image.at<cv::Vec3b>(i, j)[2];

			if (B < 32) {
				data.at<cv::Vec3i>(0, 0)[0]++;
			}
			else if (B < 64) {
				data.at<cv::Vec3i>(0, 1)[0]++;
			}
			else if (B < 96) {
				data.at<cv::Vec3i>(0, 2)[0]++;
			}
			else if (B < 128) {
				data.at<cv::Vec3i>(0, 3)[0]++;
			}
			else if (B < 160) {
				data.at<cv::Vec3i>(0, 4)[0]++;
			}
			else if (B < 192) {
				data.at<cv::Vec3i>(0, 5)[0]++;
			}
			else if (B < 224) {
				data.at<cv::Vec3i>(0, 6)[0]++;
			}
			else {
				data.at<cv::Vec3i>(0, 7)[0]++;
			}

			if (G < 32) {
				data.at<cv::Vec3i>(0, 0)[1]++;
			}
			else if (G < 64) {
				data.at<cv::Vec3i>(0, 1)[1]++;
			}
			else if (G < 96) {
				data.at<cv::Vec3i>(0, 2)[1]++;
			}
			else if (G < 128) {
				data.at<cv::Vec3i>(0, 3)[1]++;
			}
			else if (G < 160) {
				data.at<cv::Vec3i>(0, 4)[1]++;
			}
			else if (G < 192) {
				data.at<cv::Vec3i>(0, 5)[1]++;
			}
			else if (G < 224) {
				data.at<cv::Vec3i>(0, 6)[1]++;
			}
			else {
				data.at<cv::Vec3i>(0, 7)[1]++;
			}

			if (R < 32) {
				data.at<cv::Vec3i>(0, 0)[2]++;
			}
			else if (R < 64) {
				data.at<cv::Vec3i>(0, 1)[2]++;
			}
			else if (R < 96) {
				data.at<cv::Vec3i>(0, 2)[2]++;
			}
			else if (R < 128) {
				data.at<cv::Vec3i>(0, 3)[2]++;
			}
			else if (R < 160) {
				data.at<cv::Vec3i>(0, 4)[2]++;
			}
			else if (R < 192) {
				data.at<cv::Vec3i>(0, 5)[2]++;
			}
			else if (R < 224) {
				data.at<cv::Vec3i>(0, 6)[2]++;
			}
			else {
				data.at<cv::Vec3i>(0, 7)[2]++;
			}
		}
	}
}

/**
 * 根据统计数据画出颜色直方图
 * srcImage [INPUT] 产生data 的源图像，即data 的数据来源
 *                  用于计算总像素数量，确定y 轴数量级
 */
void visualColorData(cv::Mat &srcImage, cv::Mat& data, cv::Mat& image)
{
	image.setTo(0);

	// 把image 分成上、中、下三部分，分别显示R G B的颜色直方图
	cv::Mat rHeader, gHeader, bHeader;
	
	rHeader = image.rowRange(0, 200);
	gHeader = image.rowRange(200, 400);
	bHeader = image.rowRange(400, 600);

	int pixCount = srcImage.rows * srcImage.cols;
	int yUnit = 0;

	assert(pixCount > 0);

	yUnit = pixCount / 200 + 1; // y 轴每像素代表的数量值

	cv::Point ptLB; // 左下
	cv::Point ptLT; // 左上
	cv::Point ptRB; // 右下

	for (int i = 0; i < data.rows; i++) {
		for (int j = 0; j < data.cols; j++) {
			int B = data.at<cv::Vec3i>(i, j)[0];
			int G = data.at<cv::Vec3i>(i, j)[1];
			int R = data.at<cv::Vec3i>(i, j)[2];

			//ptLT.x = 10;
			//ptLT.y = 20;
			//ptRB.x = 30;
			//ptRB.y = 230;
			// 经测试opencv 原点为左上


			// 画R 直方图
			if (R > 0) {
				// 直方图高
				int h = R / yUnit;
				ptLT.y = 200 - h;
				ptLT.x = j * 32;
				ptRB.y = 200;
				ptRB.x = (j+1) * 32;
				cv::rectangle(rHeader, ptLT, ptRB, cv::Scalar(0, 0, 255), cv::FILLED);
			}

			// 画G 直方图
			if (G > 0) {
				// 直方图高
				int h = G / yUnit;
				ptLT.y = 200 - h;
				ptLT.x = j * 32;
				ptRB.y = 200;
				ptRB.x = (j + 1) * 32;
				cv::rectangle(gHeader, ptLT, ptRB, cv::Scalar(0, 255, 0), cv::FILLED);
			}

			// 画B 直方图
			if (B > 0) {
				// 直方图高
				int h = B / yUnit;
				ptLT.y = 200 - h;
				ptLT.x = j * 32;
				ptRB.y = 200;
				ptRB.x = (j + 1) * 32;
				cv::rectangle(bHeader, ptLT, ptRB, cv::Scalar(255, 0, 0), cv::FILLED);
			}
		}
	}
}
/*
int main()
{
	help();

	box = cv::Rect(-1, -1, 0, 0);
	cv::Mat image, frontend;
	//cv::Mat image(200, 200, CV_8UC3);
	image = cv::imread("D:\\work\\SV\\1.bmp");
	image.copyTo(frontend);
	//highlightRemove(frontend, frontend);

	// 三通道256/32=8长度的一维数组，存放颜色分量统计数据
	cv::Mat histoData = cv::Mat::zeros(1, 8, CV_32SC3);

	// histoImage 分成上、中、下三部分，分别显示R G B的颜色直方图
	cv::Mat histoImage;// , rHeader, gHeader, bHeader;
	histoImage = cv::Mat::zeros(600, 256, CV_8UC3);
	//rHeader = histoImage.rowRange(0, 200);
	//gHeader = histoImage.rowRange(200, 400);
	//bHeader = histoImage.rowRange(400, 600);

	cv::Mat* params[4] = { &image, &frontend, &histoData, &histoImage };

	//box = cv::Rect(-1, -1, 0, 0);
	//image = cv::Scalar::all(0);

	cv::namedWindow("Exercise8-3");
	cv::namedWindow("histogram");

	cv::imshow("histogram", histoImage);

	//cv::setMouseCallback("Exercise8-3", my_mouse_callback, (void*)& temp);
	cv::setMouseCallback("Exercise8-3", my_mouse_callback, (void*)params);

	for (;;) {
		if (drawing_box) {
			image.copyTo(frontend);
			draw_box(frontend, box);
		}

		cv::imshow("Exercise8-3", frontend);
		cv::imshow("histogram", histoImage);

		if (cv::waitKey(15) == 27) {
			break;
		}
	}

	return 0;
}

void my_mouse_callback(int event, int x, int y, int flags, void* param)
{
	cv::Mat& src = *(*(cv::Mat * *)param);
	cv::Mat& frontend = *(*((cv::Mat * *)param + 1));
	cv::Mat& histoData = *(*((cv::Mat * *)param+2));
	cv::Mat& histoImage = *(*((cv::Mat * *)param+3));

	cv::Mat selectedFront, selectedSrc;

	switch (event) {
	case cv::EVENT_MOUSEMOVE:
		if (drawing_box) {
			box.width = x - box.x;
			box.height = y - box.y;
		}
		break;

	case cv::EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = cv::Rect(x, y, 0, 0);
		break;

	case cv::EVENT_LBUTTONUP:
		drawing_box = false;
		if (box.width < 0) {
			box.x += box.width;
			box.width *= -1;
		}

		if (box.height < 0) {
			box.y += box.height;
			box.height *= -1;
		}
		draw_box(frontend, box);
		if (box.width > 0 && box.height > 0) {
			// 勉强算高亮？？
			selectedFront = 
				frontend.rowRange(box.y, box.y + box.height).colRange(box.x, box.x + box.width);
			cv::bitwise_not(selectedFront, selectedFront);

			selectedSrc =
				src.rowRange(box.y, box.y + box.height).colRange(box.x, box.x + box.width);
		}
		
		// 统计选中区域颜色信息，显示直方图
		statisticsColor(selectedSrc, histoData);
		visualColorData(selectedSrc, histoData, histoImage);
		break;
	}
}
*/
/*
// Exercise 8-4
void onTrackbarSlide(int pos, void*) {
	//
	//std::cout << "g_slider_position " << g_slider_position << std::endl;
}

int main(int argc, char** argv) {
	//std::cout << argv[1] << std::endl;
	int slider_position = 0;
	int on_off = 0;

	cv::namedWindow("Exercise8-4", cv::WINDOW_AUTOSIZE);

	cv::VideoCapture cap;
	//cap.open(cv::String(argv[1]));
	cap.open("D:\\work\\SV\\SampleVideo_1280x720_30mb.mp4");

	double frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
	double fps_stand = cap.get(cv::CAP_PROP_FPS);
	double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "frames: " << (int)frames << " fps: " << fps_stand 
		<< " w: " << width << " h: " << height << std::endl;

	cv::Mat frame;

	cv::createTrackbar("Position", "Exercise8-4", &slider_position, frames, NULL);
	cv::createTrackbar("OnOff", "Exercise8-4", &on_off, 1, NULL);

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
	int current_pos = 0;
	for (;;) {
		if (on_off) {
			cv::waitKey(50);
			continue;
		}
		cap >> frame;
		if (frame.empty()) {
			break;
		}

		// 检查滑动条实际位置，如果与当前播放位置相差过大就调整
		int slider_pos = cv::getTrackbarPos("Position", "Exercise8-4");
		if (abs(slider_pos - current_pos) > 10) {
			cap.set(cv::CAP_PROP_POS_FRAMES, slider_pos);
			current_pos = slider_pos;
			continue;
		}

		current_pos = (int)cap.get(cv::CAP_PROP_POS_FRAMES);
		cv::setTrackbarPos("Position", "Exercise8-4", current_pos);

		i64Cur = cv::getTickCount();
		if (i64Count > 0) {
			//dFrq = cv::getTickFrequency();
			//std::cout << "dFrq: " << dFrq << std::endl;
			fps = (double)i64Count / ((i64Cur - i64Begin) / dFrq);
			if (fps < fps_stand) {
				delay--;
			}
			else if (fps > fps_stand) {
				delay++;
			}
			if (0 == delay) {
				delay++;
			}
			//std::cout << "delay: " << delay << std::endl;
			srand(time(0));

			sprintf_s(buf, "fps: %f", fps);
			text.clear();
			text.append(buf);

			cv::putText(frame, text, ori,
				cv::FONT_HERSHEY_SIMPLEX, 1, color);
		}
		i64Count++;

		cv::imshow("Exercise8-4", frame);

		if (cv::waitKey(delay) >= 0) {
			break;
		}
	}

	return 0;
}
*/
/*
// Exercise 8-5
// 没太理解b 部分的要求，改为对选中区域进行逻辑操作
cv::Rect box;
bool drawing_box = false;
bool erasing = false;
int g_switcher = 0;
int g_logic = 0;

void draw_box(cv::Mat& img, cv::Rect& box)
{
	switch (g_switcher) {
	case 0:
		cv::rectangle(img, box.tl(), box.br(), cv::Scalar(0, 0, 255));
		break;

	case 1:
	{
		cv::Point center((box.tl().x + box.br().x) / 2,
			(box.tl().y + box.br().y) / 2);
		cv::Size axes(abs(box.width) / 2, abs(box.height) / 2);
		cv::ellipse(img, center, axes, 0, 0, 360, cv::Scalar(0, 255, 0));
	}
		break;

	case 2:
		cv::line(img, box.tl(), box.br(), cv::Scalar(255, 0, 0));
		break;
	}
}

void onTrackbarSlide(int pos, void*) {
	//
	std::cout << "g_slider_position " << g_switcher << std::endl;
}

void cb_mouse_rectangle(int event, int x, int y, int flags, void* param)
{
	//cv::Mat& image = *(cv::Mat *)param;
	cv::Mat& src = *(*(cv::Mat * *)param);
	cv::Mat& frontend = *(*((cv::Mat * *)param + 1));

	switch (event) {
	case cv::EVENT_MOUSEMOVE:
		if (drawing_box) {
			box.width = x - box.x;
			box.height = y - box.y;
		}
		else if (erasing) {
			frontend.at<cv::Vec3b>(y, x)[0] = 0;
			frontend.at<cv::Vec3b>(y, x)[1] = 0;
			frontend.at<cv::Vec3b>(y, x)[2] = 0;
		}
		else if (g_logic && abs(box.width) > 0 && abs(box.height) > 0) {
			cv::Mat selectedFront =
				frontend.rowRange(box.y, box.y + box.height).colRange(box.x, box.x + box.width);
			if (1 == g_logic) {
				cv::bitwise_and(selectedFront, selectedFront, selectedFront);
			}
			else if (2 == g_logic) {
				cv::bitwise_or(selectedFront, selectedFront, selectedFront);
			}
			else if (3 == g_logic) {
				cv::bitwise_not(selectedFront, selectedFront);
			}
			
		}
		break;

	case cv::EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = cv::Rect(x, y, 0, 0);
		break;

	case cv::EVENT_LBUTTONUP:
		drawing_box = false;
		if (box.width < 0) {
			box.x += box.width;
			box.width *= -1;
		}

		if (box.height < 0) {
			box.y += box.height;
			box.height *= -1;
		}
		draw_box(frontend, box);
		frontend.copyTo(src); // 写入源图像中
		break;

	case cv::EVENT_RBUTTONDOWN:
		erasing = true;
		frontend.at<cv::Vec3b>(y, x)[0] = 0;
		frontend.at<cv::Vec3b>(y, x)[1] = 0;
		frontend.at<cv::Vec3b>(y, x)[2] = 0;
		break;

	case cv::EVENT_RBUTTONUP:
		erasing = false;
		frontend.at<cv::Vec3b>(y, x)[0] = 0;
		frontend.at<cv::Vec3b>(y, x)[1] = 0;
		frontend.at<cv::Vec3b>(y, x)[2] = 0;
		frontend.copyTo(src); // 写入源图像中
		break;
	}

}

int main(int argc, char** argv) {
	//std::cout << argv[1] << std::endl;
	int slider_position = 0;
	
	cv::Mat image = cv::Mat::zeros(400, 400, CV_8UC3);
	cv::Mat frontend = cv::Mat::zeros(400, 400, CV_8UC3);

	cv::Mat* params[2] = { &image, &frontend};

	cv::namedWindow("Exercise8-5", cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("Switcher", "Exercise8-5", &g_switcher, 2, onTrackbarSlide);
	cv::createTrackbar("LogicalDrawing", "Exercise8-5", &g_logic, 3, onTrackbarSlide);

	cv::imshow("Exercise8-5", image);

	//cv::setMouseCallback("Exercise8-5", cb_mouse_rectangle, (void*)&image);
	cv::setMouseCallback("Exercise8-5", cb_mouse_rectangle, (void*)& params);

	for (;;) {
		if (drawing_box) {
			image.copyTo(frontend);
			draw_box(frontend, box);
		}
		else if (erasing) {

		}

		cv::imshow("Exercise8-5", frontend);

		if (cv::waitKey(15) == 27) {
			break;
		}
	}

	return 0;
}
*/
/*
// Exercise 8-6
// 在Exercise 4-1 的基础上修改而成
bool g_bEditing = false;
int g_startX = 0;
int g_curX = 0;
int g_curY = 0;
int g_cancel = 0;
cv::Size g_char_size;
cv::Scalar g_colorBlack(0, 0, 0);
cv::Scalar g_colorWhite(255, 255, 255);
std::vector<cv::Scalar> g_vecColors;

///////////////
// 获取字符尺寸
//
///////////////
cv::Size OutputCharWidthHeight(const cv::String& cvstr)
{
	int baseLine = 0;

	cv::Size char_size = cv::getTextSize(cvstr, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);
	std::cout << cvstr << "(w, h) is (" << char_size.width << ", " << char_size.height << ")" << std::endl;

	return char_size;
}

// 擦除背景
// 另：使用putText 写空格的方法进行背景处理是无效的
void earse(cv::Mat& img, cv::Point& cvpOrigin, const cv::Size& char_size,
	const cv::Scalar& colorBackGround)
{
	cv::Point rect4points[4];
	rect4points[0].x = cvpOrigin.x;
	rect4points[0].y = cvpOrigin.y;
	rect4points[1].x = cvpOrigin.x;
	rect4points[1].y = cvpOrigin.y - char_size.height;
	rect4points[2].x = cvpOrigin.x + char_size.width;
	rect4points[2].y = cvpOrigin.y - char_size.height;
	rect4points[3].x = cvpOrigin.x + char_size.width;
	rect4points[3].y = cvpOrigin.y;
	cv::fillConvexPoly(cv::InputOutputArray(img), rect4points, 4, colorBackGround);
}

void restore(cv::Mat& img, int x, int y, uchar old_char,
	const cv::Size& char_size, const cv::Scalar& colorBackGround,
	const cv::String& window, bool bColorful
)
{
	int curColor = 0;
	cv::String cvstr(" ");
	cv::Scalar curScalar;
	cv::Point cvpOrigin(x, y);

	earse(img, cvpOrigin, char_size, colorBackGround);
	// 重写当前字符
	curColor = abs(old_char - 0x30)%10;
	curScalar = bColorful ? g_vecColors[curColor] : g_colorWhite;
	cvstr.at(0) = old_char;
	cv::putText(img, cvstr, cvpOrigin, cv::FONT_HERSHEY_SIMPLEX, 1, curScalar);
	cv::imshow(window, img);
}

void cb_mouse(int event, int x, int y, int flags, void* param)
{
	cv::Mat& image = *(cv::Mat *)param;

	switch (event) {
	
	case cv::EVENT_LBUTTONDOWN:
		if (!g_bEditing) {
			// 进入编辑状态
			// 在鼠标位置显示光标
			g_bEditing = true;
			g_startX = x;
			g_curX = x;
			g_curY = y;
		}
		break;
	}

}

void cb_on_cancel(int state, void *params) {
	//
	std::cout << "cb_on_cancel state:" << state << std::endl;
}

void onCancel(int pos, void* params) {
	std::cout << "on_cancel begin." << std::endl;
	cv::Mat& img = *(cv::Mat*)params;
	cv::Point cvpOrigin;

	if (g_bEditing) {
		g_bEditing = false;
		//while (g_curX > g_startX) {
		//	restore(img, g_curX, g_curY, ' ', g_char_size, g_colorBlack, "Exercise8-6", true);
		//
		//	g_curX -= g_char_size.width;
		//}
		
		cvpOrigin.x = g_curX;
		cvpOrigin.y = g_curY;
		earse(img, cvpOrigin, g_char_size, g_colorBlack);
		
		cv::imshow("Exercise8-6", img);
	}
	std::cout << "on_cancel end." << std::endl;
}

int main(int argc, char** argv) {
	cv::String cvsWindow("Exercise8-6");

	const int PIXES_WIDTH = 500;
	const int PIXES_HEIGHT = 500;

	cv::Scalar color0(255, 0, 0);
	cv::Scalar color1(CV_RGB(0, 255, 0));
	cv::Scalar color2(0, 0, 255);
	cv::Scalar color3(255, 255, 0);
	cv::Scalar color4(0, 255, 255);
	cv::Scalar color5(255, 0, 255);
	cv::Scalar color6(255, 255, 255);
	cv::Scalar color7(128, 0, 0);
	cv::Scalar color8(128, 128, 0);
	cv::Scalar color9(128, 128, 128);

	g_vecColors.push_back(color0);
	g_vecColors.push_back(color1);
	g_vecColors.push_back(color2);
	g_vecColors.push_back(color3);
	g_vecColors.push_back(color4);
	g_vecColors.push_back(color5);
	g_vecColors.push_back(color6);
	g_vecColors.push_back(color7);
	g_vecColors.push_back(color8);
	g_vecColors.push_back(color9);

	std::vector<cv::String> vecChars;

	vecChars.push_back(cv::String("0"));
	vecChars.push_back(cv::String("1"));
	vecChars.push_back(cv::String("2"));
	vecChars.push_back(cv::String("3"));
	vecChars.push_back(cv::String("4"));
	vecChars.push_back(cv::String("5"));
	vecChars.push_back(cv::String("6"));
	vecChars.push_back(cv::String("7"));
	vecChars.push_back(cv::String("8"));
	vecChars.push_back(cv::String("9"));
	//vecChars.push_back(cv::String("A"));
	//vecChars.push_back(cv::String("a"));
	//vecChars.push_back(cv::String("人"));

	int iCount = 0;
	int baseLine = 0;
	int input_char = 0;
	int old_char = 0;
	int curRow = 0;
	int curCol = 0;
	int curColor = 0;
	bool bColorful = true;

	cv::Size char_size;
	for (std::vector<cv::String>::iterator it = vecChars.begin(); it != vecChars.end(); it++) {
		char_size = OutputCharWidthHeight(*it);
		g_char_size = char_size;
	}

	cv::Point cvpOrigin(-char_size.width, char_size.height);

	const int ROWS = PIXES_HEIGHT / char_size.height;
	const int COLS = PIXES_WIDTH / char_size.width;
	cv::Mat text = cv::Mat::zeros(ROWS, COLS, CV_8UC1);

	cv::Mat img = cv::Mat::zeros(500, 500, CV_8UC3);

	if (img.empty()) {
		return -1;
	}

	cv::namedWindow(cvsWindow, cv::WINDOW_AUTOSIZE);
	cv::imshow(cvsWindow, img);

	cv::String cvstrBlank(" ");
	cv::String cvstrCursor("I");
	cv::String cvstr;

	cvstr.push_back('0');

	cv::setMouseCallback("Exercise8-6", cb_mouse, (void*)&img);

	//cv::createButton("cancel", cb_on_cancel);// need QT support
	cv::createTrackbar("cancel", "Exercise8-6", &g_cancel, 1, onCancel, &img);

	while (true) {

		if (g_bEditing) {
			// 当前光标处背景擦除
			if (0 == (iCount / 10) % 2) {
				cvpOrigin.x = g_curX;
				cvpOrigin.y = g_curY;
				earse(img, cvpOrigin, char_size, g_colorBlack);
				cv::imshow(cvsWindow, img);
			}
			input_char = cv::waitKeyEx(50);

			if (27 == input_char) {
				break;
			}

			// 数字键，在当前光标处输出数字，然后光标前进

			// 方向键，移动光标

			// 回车

			// 退格

			// 颜色控制

			// 其他键，忽略

			//if (input_char >= 0x30 && input_char <= 0x39) {
			if (input_char >= 0x30 && input_char <= 0x7A) {
				curColor = (input_char - 0x30)%10;

				// 擦除光标
				cvpOrigin.x = g_curX;
				cvpOrigin.y = g_curY;
				earse(img, cvpOrigin, char_size, g_colorBlack);

				cvstr.at(0) = input_char;
				text.at<uchar>(curRow, curCol) = input_char;
				cv::putText(img, cvstr, cvpOrigin, cv::FONT_HERSHEY_SIMPLEX, 1, g_vecColors[curColor]);
				cv::imshow(cvsWindow, img);

				if (g_curX < COLS*char_size.width) {
					g_curX += char_size.width;
				}
			}
			else if (0x08 == input_char) { // 退格
				// 
				restore(img, g_curX, g_curY, ' ', char_size, g_colorBlack, cvsWindow, bColorful);

				if (g_curX > g_startX) {
					g_curX -= char_size.width;
				}
			}
			else if (0xd == input_char) {
				// 回车
				// 擦除光标
				cvpOrigin.x = g_curX;
				cvpOrigin.y = g_curY;
				earse(img, cvpOrigin, char_size, g_colorBlack);
				cv::imshow(cvsWindow, img);
				g_bEditing = false;
				continue;
			}
			
			// 显示光标
			// 注意后面的g_bEditing，
			// 滑动条事件会在休眠期间发生，这里必须再判断一下g_bEditing
			if (1 == (iCount / 10) % 2 && g_bEditing) {
				cvpOrigin.x = g_curX;;
				cvpOrigin.y = g_curY;
				cv::putText(img, cvstrCursor, cvpOrigin, cv::FONT_HERSHEY_SIMPLEX, 1, g_colorWhite);
				cv::imshow("Exercise8-6", img);
			}
			std::cout << "main show cursor." << std::endl;
			iCount++;
		}
		else {
			input_char = cv::waitKeyEx(50);
		}
		
	}

	cv::destroyWindow("Exercise8-6");

	return 0;
}
*/
/*
// Exercise 8-7
// 参考Example 11-2、Example 11-1
int local_convert(int input_char)
{
	int ret = 0;
	switch (input_char) {
	case '!':
		ret = 0x21;
		break;
	case '@':
		ret = 0x22;
		break;
	case '#':
		ret = 0x23;
		break;
	case '$':
		ret = 0x24;
		break;
	case '%':
		ret = 0x25;
		break;
	case '^':
		ret = 0x26;
		break;
	case '&':
		ret = 0x27;
		break;
	case '*':
		ret = 0x28;
		break;
	case '(':
		ret = 0x29;
		break;
	}

	return ret;
}

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

	cv::Mat warp_mat = cv::getPerspectiveTransform(srcQuad, dstQuad);
	cv::Mat dst;
	cv::warpPerspective(src, dst, warp_mat, src.size(),
		cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

	int type = warp_mat.type();
	std::cout << "warp_mat type: " << type << std::endl;

	int channels = warp_mat.channels();
	std::cout << "warp_mat channels: " << channels << std::endl;

	cv::Size size = warp_mat.size();
	std::cout << "warp_mat size: " << channels << std::endl;

	for (int i = 0; i < size.height; i++) {
		for (int j = 0; j < size.width; j++) {
			std::cout << "warp_mat[" << i << "," << j << "]" 
				<< warp_mat.at<double>(i, j) << std::endl;
		}
	}
	
	for (int i = 0; i < 4; i++) {
		cv::circle(dst, dstQuad[i], 5, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
	}

	cv::imshow("Perspective Transform Test", dst);

	double curScale = 1.0;
	cv::Mat scaleDst;
	cv::namedWindow("scale", cv::WINDOW_AUTOSIZE);
	cv::imshow("scale", src);

	double rotateScale = 0.5;
	double rotateAngle = 0;
	cv::Point2f rotateCenter(src.cols * 0.5f, src.rows * 0.5f);
	cv::Mat rotateDst;
	cv::namedWindow("rotate", cv::WINDOW_AUTOSIZE);
	cv::imshow("rotate", src);

	int input_char = 0;
	int index = 0;
	while (true) {
		input_char = cv::waitKeyEx(50);
		//std::cout << "input_char: " << input_char << std::endl;
		if (27 == input_char) {
			break;
		}
		else if (input_char >= 0x31 && input_char <= 0x39) {
			index = input_char - 0x30 - 1;

			double old_value = warp_mat.at<double>(index / 3, index % 3);
			double new_value = old_value + old_value * 0.1;
			int rIndex = index / 3;
			int cIndex = index % 3;
			warp_mat.at<double>(rIndex, cIndex) = new_value;
			std::cout << "warp_mat[" << rIndex << "," << cIndex << "]"
				<< warp_mat.at<double>(rIndex, cIndex) << std::endl;
			cv::warpPerspective(src, dst, warp_mat, src.size(),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

			for (int i = 0; i < 4; i++) {
				cv::circle(dst, dstQuad[i], 5, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
			}

			cv::imshow("Perspective Transform Test", dst);
		}
		else if (local_convert(input_char) >= 0x21 && local_convert(input_char) <= 0x29) {
			int convert_char = local_convert(input_char);
			index = convert_char - 0x20 - 1;

			double old_value = warp_mat.at<double>(index / 3, index % 3);
			double new_value = old_value - old_value * 0.1;
			int rIndex = index / 3;
			int cIndex = index % 3;
			warp_mat.at<double>(rIndex, cIndex) = new_value;
			std::cout << "warp_mat[" << rIndex << "," << cIndex << "]"
				<< warp_mat.at<double>(rIndex, cIndex) << std::endl;
			cv::warpPerspective(src, dst, warp_mat, src.size(),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

			for (int i = 0; i < 4; i++) {
				cv::circle(dst, dstQuad[i], 5, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
			}

			cv::imshow("Perspective Transform Test", dst);
		}
		else if (0x2B == input_char || 0x2D == input_char) { // 2b(+) 2d(-)
			// 缩放
			if (0x2B == input_char) {
				curScale += 0.1;
				if (curScale > 10.0) {
					curScale = 10.0;
				}
			}
			else if (0x2D == input_char) {
				curScale -= 0.1;
				if (curScale < 0.1) {
					curScale = 0.1;
				}
			}
			cv::resize(src, scaleDst, cv::Size(0, 0), curScale, curScale);
			cv::imshow("scale", scaleDst);
		}
		else if (0x250000 == input_char || 0x270000 == input_char) {
			// rotate
			if (0x250000 == input_char) {
				rotateAngle += 10;
			}
			else {
				rotateAngle -= 10;
			}
			
			cv::Mat rot_mat = cv::getRotationMatrix2D(rotateCenter, rotateAngle, rotateScale);

			cv::warpAffine(src, rotateDst, rot_mat, src.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

			cv::imshow("rotate", rotateDst);
		}
	}
	
	return 0;
}
*/
/*
// Exercise 8-8
using namespace std;
using namespace cv;

static void help()
{
	cout << "\nThis program demonstrates the use of cv::CascadeClassifier class to detect objects (Face + eyes). You can use Haar or LBP features.\n"
		"This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
		"It's most known use is for faces.\n"
		"Usage:\n"
		"./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
		"   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
		"   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
		"   [--try-flip]\n"
		"   [filename|camera_index]\n\n"
		"see facedetect.cmd for one call:\n"
		"./facedetect --cascade=\"data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"data/haarcascades/haarcascade_eye_tree_eyeglasses.xml\" --scale=1.3\n\n"
		"During execution:\n\tHit any key to quit.\n"
		"\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip, Mat &skull, Mat& frontend, double beta);

string cascadeName;
string nestedCascadeName;

void onTrackbarSlide(int pos, void *params) {
	//
	cout << "pos: " << pos << endl;
	Mat *pimg = *((Mat * *)params);
	CascadeClassifier* pcascade = *((CascadeClassifier * *)params+1);
	CascadeClassifier* pnestedCascade = *((CascadeClassifier * *)params + 2);
	double *pscale = *((double **)params+3);
	bool *ptryflip = *((bool**)params + 4);
	Mat* pskull = *((Mat * *)params+5);
	Mat* pfrontend = *((Mat * *)params + 6);

	double beta = pos / 10.0;
	detectAndDraw(*pimg, *pcascade, *pnestedCascade, *pscale, *ptryflip, *pskull, *pfrontend, beta);
}

int main(int argc, const char** argv)
{
	VideoCapture capture;
	Mat frame, image, skull, frontend;
	string inputName;
	bool tryflip;
	CascadeClassifier cascade, nestedCascade;
	double scale;

	int slider_position = 0;

	cv::CommandLineParser parser(argc, argv,
		"{help h||}"
		"{cascade|data/haarcascades/haarcascade_frontalface_alt.xml|}"
		"{nested-cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}"
		"{scale|1|}{try-flip||}{@filename||}"
	);
	if (parser.has("help"))
	{
		help();
		return 0;
	}
	cascadeName = parser.get<string>("cascade");
	nestedCascadeName = parser.get<string>("nested-cascade");
	scale = parser.get<double>("scale");
	if (scale < 1)
		scale = 1;
	tryflip = parser.has("try-flip");
	inputName = parser.get<string>("@filename");
	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}
	if (!nestedCascade.load(samples::findFileOrKeep(nestedCascadeName)))
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
	if (!cascade.load(samples::findFile(cascadeName)))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		help();
		return -1;
	}
	if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1))
	{
		int camera = inputName.empty() ? 0 : inputName[0] - '0';
		if (!capture.open(camera))
		{
			cout << "Capture from camera #" << camera << " didn't work" << endl;
			return 1;
		}
	}
	else if (!inputName.empty())
	{
		image = imread(samples::findFileOrKeep(inputName), IMREAD_COLOR);
		if (image.empty())
		{
			if (!capture.open(samples::findFileOrKeep(inputName)))
			{
				cout << "Could not read " << inputName << endl;
				return 1;
			}
		}
	}
	else
	{
		image = imread(samples::findFile("lena.jpg"), IMREAD_COLOR);
		if (image.empty())
		{
			cout << "Couldn't read lena.jpg" << endl;
			return 1;
		}
	}

	cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
	skull = imread(samples::findFile("skull.jpg"), IMREAD_COLOR);
	if (skull.empty()) {
		cout << "Couldn't read skull.jpg" << endl;
		return 1;
	}

	image.copyTo(frontend);

	void* params[] = {&image, &cascade, &nestedCascade, &scale, &tryflip, &skull, &frontend};
	cv::createTrackbar("alpha-beta", "result", &slider_position, 10, onTrackbarSlide, &params);

	if (capture.isOpened())
	{
		cout << "Video capturing has been started ..." << endl;

		for (;;)
		{
			capture >> frame;
			if (frame.empty())
				break;

			Mat frame1 = frame.clone();
			detectAndDraw(frame1, cascade, nestedCascade, scale, tryflip, skull, frontend, 1);

			char c = (char)waitKey(10);
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}
	}
	else
	{
		cout << "Detecting face(s) in " << inputName << endl;
		if (!image.empty())
		{
			detectAndDraw(image, cascade, nestedCascade, scale, tryflip, skull, frontend, 1);
			waitKey(0);
		}
		else if (!inputName.empty())
		{
			// assume it is a text file containing the
			// list of the image filenames to be processed - one per line 
			FILE* f = NULL;
			errno_t err = fopen_s(&f, inputName.c_str(), "rt");
			if (f)
			{
				char buf[1000 + 1];
				while (fgets(buf, 1000, f))
				{
					int len = (int)strlen(buf);
					while (len > 0 && isspace(buf[len - 1]))
						len--;
					buf[len] = '\0';
					cout << "file " << buf << endl;
					image = imread(buf, 1);
					if (!image.empty())
					{
						detectAndDraw(image, cascade, nestedCascade, scale, tryflip, skull, frontend, 1);
						char c = (char)waitKey(0);
						if (c == 27 || c == 'q' || c == 'Q')
							break;
					}
					else
					{
						cerr << "Aw snap, couldn't read image " << buf << endl;
					}
				}
				fclose(f);
			}
		}
	}

	return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip, Mat &skull, Mat& frontend, double beta)
{
	double t = 0;
	vector<Rect> faces, faces2;
	const static Scalar colors[] =
	{
		Scalar(255,0,0),
		Scalar(255,128,0),
		Scalar(255,255,0),
		Scalar(0,255,0),
		Scalar(0,128,255),
		Scalar(0,255,255),
		Scalar(0,0,255),
		Scalar(255,0,255)
	};
	Mat gray, smallImg, skullCopy;

	img.copyTo(frontend);

	cvtColor(img, gray, COLOR_BGR2GRAY);
	double fx = 1 / scale;
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
	equalizeHist(smallImg, smallImg);

	t = (double)getTickCount();
	cascade.detectMultiScale(smallImg, faces,
		1.1, 2, 0
		//|CASCADE_FIND_BIGGEST_OBJECT
		//|CASCADE_DO_ROUGH_SEARCH
		| CASCADE_SCALE_IMAGE,
		Size(30, 30));
	if (tryflip)
	{
		flip(smallImg, smallImg, 1);
		cascade.detectMultiScale(smallImg, faces2,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			| CASCADE_SCALE_IMAGE,
			Size(30, 30));
		for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r)
		{
			faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
		}
	}
	t = (double)getTickCount() - t;
	printf("detection time = %g ms\n", t * 1000 / getTickFrequency());
	for (size_t i = 0; i < faces.size(); i++)
	{
		Rect r = faces[i];
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[i % 8];
		int radius;

		double aspect_ratio = (double)r.width / r.height;

		cv::resize(skull, skullCopy, r.size());

		// this is for exercise 8.a
		//skull.copyTo(img.rowRange(r.y, r.y + r.height).colRange(r.x, r.x + r.width));

		// this is for exercise 8.b
		cv::addWeighted(frontend.rowRange(r.y, r.y + r.height).colRange(r.x, r.x + r.width), 1.0-beta, 
			skullCopy, beta, 0.5, frontend.rowRange(r.y, r.y + r.height).colRange(r.x, r.x + r.width));
			
		if (nestedCascade.empty())
			continue;
		smallImgROI = smallImg(r);
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			//|CASCADE_DO_CANNY_PRUNING
			| CASCADE_SCALE_IMAGE,
			Size(30, 30));
		for (size_t j = 0; j < nestedObjects.size(); j++)
		{
			Rect nr = nestedObjects[j];
			center.x = cvRound((r.x + nr.x + nr.width * 0.5) * scale);
			center.y = cvRound((r.y + nr.y + nr.height * 0.5) * scale);
			radius = cvRound((nr.width + nr.height) * 0.25 * scale);
			//circle(img, center, radius, color, 3, 8, 0);
		}
	}
	imshow("result", frontend);
}
*/

// Exercise 8-9
using namespace cv;
using namespace std;

static void help()
{
	// print a welcome message, and the OpenCV version
	cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
		"Using OpenCV version " << CV_VERSION << endl;
	cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
	cout << "\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - auto-initialize tracking\n"
		"\tc - delete all the points\n"
		"\tn - switch the \"night\" mode on/off\n"
		"To add/remove a feature point click it\n" << endl;
}

Point2f point;
bool addRemovePt = false;

static void onMouse(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		point = Point2f((float)x, (float)y);
		addRemovePt = true;
	}
}

int main(int argc, char** argv)
{
	VideoCapture cap;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;

	help();
	cv::CommandLineParser parser(argc, argv, "{@input|0|}");
	string input = parser.get<string>("@input");

	if (input.size() == 1 && isdigit(input[0]))
		cap.open(input[0] - '0');
	else
		cap.open(input);

	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}

	namedWindow("LK Demo", 1);
	setMouseCallback("LK Demo", onMouse, 0);

	Mat gray, prevGray, image, frame, background;
	vector<Point2f> points[2];

	// 原程序是从摄像头或视频读取
	// 这里改为从111.bmp、112.bmp、113.bmp、114.bmp 几张图片读取
	// 这些图片是同一场景下，有些抖动
	cv::String path = "D:\\work\\helloOpenCV\\chap08\\11";
	cv::String ext = ".bmp";
	cv::String fileName = "";
	char buf[8] = { 0 };
	int no = 0;
	int rStart = 0;
	int rStop = 0;
	int cStart = 0;
	int cStop = 0;

	double diffX = 0;
	double diffY = 0;

	background = cv::Mat::zeros(1000, 1000, CV_8UC3);

	for (;;)
	{
		// note: 调整此数字切换2/3/4 张图片效果
		no = no % 3 + 1;
		fileName.clear();
		fileName.append(path);
		_itoa_s(no, buf, 10);
		fileName.append(buf);
		fileName.append(ext);
		std::cout << "fileName: " << fileName << std::endl;
		frame = cv::imread(fileName, 1);
		
		frame.copyTo(image);
		cvtColor(image, gray, COLOR_BGR2GRAY);

		if (nightMode)
			image = Scalar::all(0);

		if (needToInit)
		{
			// automatic initialization
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
			cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			addRemovePt = false;
		}
		else if (!points[0].empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
				3, termcrit, 0, 0.001);
			diffX = points[1][0].x - points[0][0].x;
			diffY = points[1][0].y - points[0][0].y;
			std::cout << "points[0][0]: " << points[0][0] 
				<< "\tpoints[1][0]:" << points[1][0] 
				<< "\tdiffX: " << diffX << " diffY: " << diffY
				<< std::endl;
			
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				if (addRemovePt)
				{
					if (norm(point - points[1][i]) <= 5)
					{
						addRemovePt = false;
						continue;
					}
				}

				if (!status[i])
					continue;

				points[1][k++] = points[1][i];
				circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
			points[1].resize(k);
		}

		if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
		{
			vector<Point2f> tmp;
			tmp.push_back(point);
			cornerSubPix(gray, tmp, winSize, Size(-1, -1), termcrit);
			points[1].push_back(tmp[0]);
			addRemovePt = false;
		}

		needToInit = false;
		// 注意这里是"-"号，可以看出几张照片中人物人位置基本没变
		rStart = (1000 - image.rows) / 2-diffY;
		rStop = rStart + image.rows;
		cStart = (1000 - image.cols) / 2 - diffX;
		cStop = cStart + image.cols;
		image.copyTo(background.rowRange(rStart, rStop).colRange(cStart, cStop));
		imshow("LK Demo", background);

		char c = (char)waitKey(1000);
		if (c == 27)
			break;
		switch (c)
		{
		case 'r':
			needToInit = true;
			break;
		case 'c':
			points[0].clear();
			points[1].clear();
			break;
		case 'n':
			nightMode = !nightMode;
			break;
		}

		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);
	}

	return 0;
}

/*
// Exercise 8-9
typedef struct my_structTag {
	int iValue;
	cv::Point pt;
	cv::Rect rc;
} my_struct;

void write_my_struct(cv::FileStorage& fs, const std::string& name, const my_struct& ms)
{
	if (!fs.isOpened()) {
		// throw exception
		return;
	}

	fs << name << "{" << "int" << ms.iValue << "point" << ms.pt << "rect" << ms.rc << "}";
}

void read_my_struct(const cv::FileStorage &fs, const cv::FileNode &ms_node, my_struct &ms)
{
	if (!fs.isOpened()) {
		// throw exception
		return;
	}

	ms.iValue = ms_node["int"];
	ms.pt.x = ms_node["point"][0];
	ms.pt.y = ms_node["point"][1];

	ms.rc.x = ms_node["rect"][0];
	ms.rc.y = ms_node["rect"][1];
	ms.rc.width = ms_node["rect"][2];
	ms.rc.height = ms_node["rect"][3];
}

int main()
{
	cv::FileStorage fs_w("exercise8-10.xml", cv::FileStorage::WRITE);
	my_struct ms_w;
	my_struct msa_w[10];
	char szNo[8] = { 'x', 0, 0, 0, 0, 0, 0, 0 };

	ms_w.iValue = 10;
	ms_w.pt = cv::Point(66, 88);
	ms_w.rc = cv::Rect(60, 80, 100, 200);

	write_my_struct(fs_w, "sky", ms_w);
	
	for (int i = 0; i < 10; i++) {
		msa_w[i].iValue = rand();
		msa_w[i].pt = cv::Point(rand(), rand());
		msa_w[i].rc = cv::Rect(rand(), rand(), rand(), rand());
		_itoa_s(i, &szNo[1], 6, 10);
		write_my_struct(fs_w, szNo, msa_w[i]);
	}

	fs_w.release();

	cv::FileStorage fs_r("exercise8-10.xml", cv::FileStorage::READ);
	my_struct ms_r;

	cv::FileNode fn;
	fn = fs_r["sky"];
	read_my_struct(fs_r, fn, ms_r);
	std::cout << "read result: " << ms_r.iValue << " " << ms_r.pt << " " << ms_r.rc << std::endl;

	for (int j = 0; j < 10; j++) {
		_itoa_s(j, &szNo[1], 6, 10);
		fn = fs_r[szNo];
		read_my_struct(fs_r, fn, ms_r);
		std::cout << szNo << ": " << ms_r.iValue << " " << ms_r.pt << " " << ms_r.rc << std::endl;
	}

	return 0;
}
*/
// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
