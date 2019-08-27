#include <windows.h>
#include <opencv2\highgui.hpp>
#include <opencv2\videoio.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\video.hpp>
#include <iostream>


// 显示一幅图片
int main(int argc, char** argv) {
	std::cout << argv[1] << std::endl;

	cv::Mat img = cv::imread(argv[1], -1);

	if (img.empty()) {
		return -1;
	}

	cv::namedWindow("Example1", cv::WINDOW_AUTOSIZE);
	cv::imshow("Example1", img);
	cv::waitKey(0);
	cv::destroyWindow("Example1");

	return 0;
}

/*
// 显示视频
int main(int argc, char** argv) {
	std::cout << argv[1] << std::endl;

	cv::namedWindow("Example3", cv::WINDOW_AUTOSIZE);
	cv::VideoCapture cap;
	cap.open(cv::String(argv[1]));
	cv::Mat frame;

	for (;;) {
		cap >> frame;
		if (frame.empty()) {
			break;
		}

		cv::imshow("Example3", frame);

		if (cv::waitKey(33) >= 0) {
			break;
		}
	}

	return 0;
}
*/
/*
// 滑动条进度控制，有bug
int g_slider_position = 0;
int g_run = 1, g_dontset = 0;
cv::VideoCapture g_cap;

CRITICAL_SECTION g_cs;

void onTrackbarSlide(int pos, void*) {
	//g_cap.set(cv::CAP_PROP_POS_FRAMES, pos);

	
	if (!g_dontset) {
		EnterCriticalSection(&g_cs);
		g_run = 1;
		g_cap.set(cv::CAP_PROP_POS_FRAMES, pos);
		LeaveCriticalSection(&g_cs);
	}

	g_dontset = 0;

}

int main(int argc, char** argv) {
	std::cout << argv[1] << std::endl;

	cv::namedWindow("Example2_4", cv::WINDOW_AUTOSIZE);
	
	g_cap.open(cv::String(argv[1]));
	int frames = (int)g_cap.get(cv::CAP_PROP_FRAME_COUNT);
	int tmpw = (int)g_cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int tmph = (int)g_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "Video has " << frames << " frames of dimensions(" 
		<< tmpw << ", " << tmph << ")." << std::endl;

	cv::createTrackbar("Position", "Example2_4", &g_slider_position, 
		frames, onTrackbarSlide);
	
	cv::Mat frame;

	if (!InitializeCriticalSectionAndSpinCount(&g_cs, 0x00000400)) {
		return 1;
	}

	for (;;) {
		if (g_run != 0) {
			g_cap >> frame;
			if (frame.empty()) {
				break;
			}

			int current_pos = (int)g_cap.get(cv::CAP_PROP_POS_FRAMES);
			g_dontset = 1;

			EnterCriticalSection(&g_cs);
			cv::setTrackbarPos("Position", "Example2_4", current_pos);
			LeaveCriticalSection(&g_cs);
			cv::imshow("Example2_4", frame);

			g_run -= 1;
		}
		
		char c = (char)cv::waitKey(10);
		if ('s' == c) {
			g_run = 1;
			std::cout << "Single step, run = " << g_run << std::endl;
		}

		if ('r' == c) {
			g_run = -1;
			std::cout << "Run mode, run = " << g_run << std::endl;
		}

		if (27 == c) {
			break;
		}
	}

	DeleteCriticalSection(&g_cs);
	
	return 0;
}
*/
/*
// 高斯模糊
void example2_5(const cv::Mat &image) {
	cv::namedWindow("Example2_5-in", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Example2_5-out", cv::WINDOW_AUTOSIZE);

	cv::imshow("Example2_5-in", image);

	cv::Mat out;

	cv::GaussianBlur(image, out, cv::Size(5, 5), 3, 3);
	for (int i = 0; i < 122; i++) {
		cv::GaussianBlur(out, out, cv::Size(5, 5), 3, 3);
	}
	
	cv::imshow("Example2_5-out", out);

	cv::waitKey(0);
}

int main(int argc, char** argv) {
	std::cout << argv[1] << std::endl;

	cv::Mat img = cv::imread(argv[1], -1);

	example2_5(img);

	return 0;
}
*/
/*
// 金字塔降采样
int main(int argc, char** argv) {
	std::cout << argv[1] << std::endl;

	cv::Mat img = cv::imread(argv[1], -1);
	cv::Mat img2;

	if (img.empty()) {
		return -1;
	}

	cv::namedWindow("Example2_6_1", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Example2_6_2", cv::WINDOW_AUTOSIZE);

	cv::imshow("Example2_6_1", img);

	cv::pyrDown(img, img2);
	cv::imshow("Example2_6_2", img2);

	cv::waitKey(0);
	cv::destroyWindow("Example2_6_1");
	cv::destroyWindow("Example2_6_2");

	return 0;
}
*/
/*
// 灰化与边缘检测器
int main(int argc, char** argv) {
	std::cout << argv[1] << std::endl;

	cv::Mat img_rgb, img_gry, img_cny;

	cv::namedWindow("Example_Rgb", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Example_Gray", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Example_Canny", cv::WINDOW_AUTOSIZE);

	img_rgb = cv::imread(argv[1], -1);
	if (img_rgb.empty()) {
		return -1;
	}

	cv::imshow("Example_Rgb", img_rgb);

	cv::cvtColor(img_rgb, img_gry, cv::COLOR_BGR2GRAY);
	cv::imshow("Example_Gray", img_gry);

	cv::Canny(img_gry, img_cny, 10, 100, 3, true);
	cv::imshow("Example_Canny", img_cny);

	cv::waitKey(0);
	cv::destroyWindow("Example_Rgb");
	cv::destroyWindow("Example_Gray");
	cv::destroyWindow("Example_Canny");

	return 0;
}
*/
/*
// 读取像素
void readPixel(int x, int y, const cv::Mat& img_rgb, const cv::Mat& img_gray,
	const cv::Mat& img_pyr) {
	cv::Vec3b intensity = img_rgb.at<cv::Vec3b>(y, x);

	uchar blue = intensity[0];
	uchar green = intensity[1];
	uchar red = intensity[2];

	std::cout << "At (x, y) = (" << x << ", " << y << "): (blue, green, red) = ("
		<< (unsigned int)blue << ", " << (unsigned int)green << ", " 
		<< (unsigned int)red << ")" << std::endl;

	std::cout << "Gray pixel there is: " 
		<< (unsigned int)img_gray.at<uchar>(y, x) << std::endl;

	x /= 4;
	y /= 4;

	std::cout << "Pyramid2 pixel there is: " 
		<< (unsigned int)img_pyr.at<uchar>(y, x) << std::endl;
}

// 串联：灰化 降采样 降采样 边缘检测
int main(int argc, char** argv) {
	std::cout << argv[1] << std::endl;

	cv::Mat img_rgb, img_gry, img_pyr, img_pyr2, img_cny;

	cv::namedWindow("Example_Rgb", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Example_Gray", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Example_Pyr", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Example_Pyr2", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Example_Canny", cv::WINDOW_AUTOSIZE);

	img_rgb = cv::imread(argv[1], -1);
	if (img_rgb.empty()) {
		return -1;
	}

	cv::imshow("Example_Rgb", img_rgb);

	cv::cvtColor(img_rgb, img_gry, cv::COLOR_BGR2GRAY);
	cv::imshow("Example_Gray", img_gry);

	cv::pyrDown(img_gry, img_pyr);
	cv::imshow("Example_Pyr", img_pyr);

	cv::pyrDown(img_pyr, img_pyr2);
	cv::imshow("Example_Pyr2", img_pyr2);

	cv::Canny(img_pyr2, img_cny, 10, 100, 3, true);
	cv::imshow("Example_Canny", img_cny);

	readPixel(16, 32, img_rgb, img_gry, img_pyr2);

	cv::waitKey(0);
	cv::destroyWindow("Example_Rgb");
	cv::destroyWindow("Example_Gray");
	cv::destroyWindow("Example_Pyr");
	cv::destroyWindow("Example_Pyr2");
	cv::destroyWindow("Example_Canny");

	return 0;
}
*/
/*
// 对数极坐标变换
int main(int argc, char **argv) {
	std::cout << argv[1] << std::endl;

	cv::namedWindow("Example2_11", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Log_Polar", cv::WINDOW_AUTOSIZE);

	cv::VideoCapture capture(argv[1]);

	double fps = capture.get(cv::CAP_PROP_FPS);
	cv::Size size(
		(int)capture.get(cv::CAP_PROP_FRAME_WIDTH), 
		(int)capture.get(cv::CAP_PROP_FRAME_HEIGHT)
	);

	cv::VideoWriter writer;
	writer.open(argv[2], cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 
		fps, size);

	cv::Mat logpolar_frame, bgr_frame;

	for (;;) {
		capture >> bgr_frame;
		if (bgr_frame.empty()) {
			break;
		}

		cv::imshow("Example2_11", bgr_frame);

		cv::logPolar(bgr_frame, logpolar_frame, 
			cv::Point2f(bgr_frame.cols / 2, bgr_frame.rows / 2), 
			40, cv::WARP_FILL_OUTLIERS);

		cv::imshow("Log_Polar", logpolar_frame);
		writer << logpolar_frame;

		char c = cv::waitKey(10);
		if (27 == c) {
			break;
		}
	}

	capture.release();

	return 0;
}
*/
/*
// 降采样，保存
int main(int argc, char** argv) {
	std::cout << argv[1] << std::endl;

	cv::namedWindow("Exercise2-3", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Pyr", cv::WINDOW_AUTOSIZE);

	cv::VideoCapture capture(argv[1]);

	double fps = capture.get(cv::CAP_PROP_FPS);
	cv::Size size(
		(int)capture.get(cv::CAP_PROP_FRAME_WIDTH)/2,
		(int)capture.get(cv::CAP_PROP_FRAME_HEIGHT)/2
	);

	cv::VideoWriter writer;
	writer.open(argv[2], cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
		fps, size);

	cv::Mat pyr_frame, bgr_frame;

	for (;;) {
		capture >> bgr_frame;
		if (bgr_frame.empty()) {
			break;
		}

		cv::imshow("Exercise2-3", bgr_frame);

		cv::pyrDown(bgr_frame, pyr_frame, size);

		cv::imshow("Pyr", pyr_frame);
		writer << pyr_frame;

		char c = cv::waitKey(10);
		if (27 == c) {
			break;
		}
	}

	capture.release();

	return 0;
}
*/
/*
// Exercise 2-5
int g_slider_position = 0;
cv::VideoCapture g_cap;

void onTrackbarSlide(int pos, void*) {
	//
	std::cout << "g_slider_position " << g_slider_position << std::endl;
}

int main(int argc, char** argv) {
	std::cout << argv[1] << std::endl;

	cv::namedWindow("Exercise2-5", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Pyr", cv::WINDOW_AUTOSIZE);

	g_cap.open(cv::String(argv[1]));
	int frames = (int)g_cap.get(cv::CAP_PROP_FRAME_COUNT);
	int tmpw = (int)g_cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int tmph = (int)g_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "Video has " << frames << " frames of dimensions("
		<< tmpw << ", " << tmph << ")." << std::endl;

	cv::createTrackbar("Position", "Exercise2-5", &g_slider_position,
		2, onTrackbarSlide);

	cv::Mat frame;
	cv::Mat frame_pyr;

	for (;;) {
		g_cap >> frame;
		if (frame.empty()) {
			break;
		}

		cv::imshow("Exercise2-5", frame);
		cv::pyrDown(frame, frame_pyr);
		for (int i = 0; i < g_slider_position; i++) {
			cv::pyrDown(frame_pyr, frame_pyr);
		}
		cv::imshow("Pyr", frame_pyr);

		char c = (char)cv::waitKey(10);
		if (27 == c) {
			break;
		}
	}

	return 0;
}
*/