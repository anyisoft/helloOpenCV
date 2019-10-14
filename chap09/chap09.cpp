// chap09.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <gl/glew.h>
#include <opencv2/core/opengl.hpp>
/*
// example 9-2
void my_mouse_callback(int event, int x, int y, int flags, void* param);

cv::Rect box;
bool drawing_box = false;

void draw_box(cv::Mat &img, cv::Rect &box)
{
	cv::rectangle(img, box.tl(), box.br(), cv::Scalar(0, 0, 255));
}

void help()
{
	std::cout << "Call: ./ch4_ext_1\n"<< 
		" shows how to use a mouse to draw regions in an image." << std::endl;
}

int main()
{
	help();

	box = cv::Rect(-1, -1, 0, 0);
	cv::Mat image(200, 200, CV_8UC3), temp;
	image = cv::imread("D:\\work\\SV\\1.bmp");
	image.copyTo(temp);

	//box = cv::Rect(-1, -1, 0, 0);
	//image = cv::Scalar::all(0);

	cv::namedWindow("Box Example");

	cv::setMouseCallback("Box Example", my_mouse_callback, (void*)& image);

	for (;;) {
		image.copyTo(temp);
		if (drawing_box) {
			draw_box(temp, box);
		}

		cv::imshow("Box Example", temp);

		if (cv::waitKey(15) == 27) {
			break;
		}
	}

	return 0;
}

void my_mouse_callback(int event, int x, int y, int flags, void* param)
{
	cv::Mat& image = *(cv::Mat*)param;

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
		draw_box(image, box);
		break;
	}
}
*/
/*
// Exercise 9-1
void resetBigImage(
	cv::Mat& imglt, cv::Mat& imgrt, cv::Mat& imglb, cv::Mat& imgrb,
	cv::Mat& img1, cv::Mat& img2, cv::Mat& img3, cv::Mat& img4);

void on_mouse(int EVENT, int x, int y, int flags, void* params)
{
	cv::Scalar color(0, 0, 255);
	cv::String text;
	
	cv::Mat imglt, imgrt, imglb, imgrb;
	cv::Mat img1, img2, img3, img4;
	cv::Mat* dst = NULL;
	cv::Mat* background;

	int showX = 0;
	int showY = 0;
	
	char buf[64] = { 0 };

	background = *((cv::Mat * *)params);
	imglt = *(*((cv::Mat * *)params + 1));
	imgrt = *(*((cv::Mat * *)params + 2));
	imglb = *(*((cv::Mat * *)params + 3));
	imgrb = *(*((cv::Mat * *)params + 4));
	img1 = *(*((cv::Mat * *)params + 5));
	img2 = *(*((cv::Mat * *)params + 6));
	img3 = *(*((cv::Mat * *)params + 7));
	img4 = *(*((cv::Mat * *)params + 8));
	cv::Point p(x, y);
	cv::Point ori;

	switch (EVENT) {

	case cv::EVENT_LBUTTONDOWN:
		// 根据坐标确定在哪个小图里面
		if (x < 300 && y < 300) {
			//
			showX = x;
			showY = y;
			dst = &imglt;
		}
		else if (x > 299 && y < 300) {
			//
			showX = x-300;
			showY = y;
			dst = &imgrt;
		}
		else if (x < 300 && y > 299) {
			//
			showX = x;
			showY = y-300;
			dst = &imglb;
		}
		else if (x > 299 && y > 299) {
			//
			showX = x-300;
			showY = y-300;
			dst = &imgrb;
		}
		
		// 在当前鼠标位置右下/左上方(20,20) 处显示当前点颜色和坐标
		sprintf_s(buf, "x=%d y=%d", showX, showY);
		text.clear();
		text.append(buf);
		ori.x = showX;
		if (ori.x > 100) {
			ori.x -= 240;
			if (ori.x < 0) {
				ori.x = 0;
			}
		}
		ori.y = showY + 20;
		if (showY > 280) {
			ori.y -= 40;
		}
		
		resetBigImage(imglt, imgrt, imglb, imgrb, img1, img2, img3, img4);
		cv::putText(*dst, text, ori,
			cv::FONT_HERSHEY_SIMPLEX, 1, color);

		cv::imshow("stage", *background);
		break;

	}
}

void resetBigImage(
	cv::Mat &imglt, cv::Mat& imgrt, cv::Mat& imglb, cv::Mat& imgrb, 
	cv::Mat& img1, cv::Mat& img2, cv::Mat& img3, cv::Mat& img4)
{
	img1.rowRange(0, 300).colRange(0, 300).copyTo(imglt);
	img2.rowRange(0, 300).colRange(0, 300).copyTo(imgrt);
	img3.rowRange(0, 300).colRange(0, 300).copyTo(imglb);
	img4.rowRange(0, 300).colRange(0, 300).copyTo(imgrb);
}

int main()
{
	cv::Mat img1, img2, img3, img4;
	cv::Mat background, imglt, imgrt, imglb, imgrb;

	background = cv::Mat::zeros(600, 600, CV_8UC3);
	imglt = background.rowRange(0, 300).colRange(0, 300);
	imgrt = background.rowRange(300, 600).colRange(0, 300);
	imglb = background.rowRange(0, 300).colRange(300, 600);
	imgrb = background.rowRange(300, 600).colRange(300, 600);

	cv::namedWindow("stage");

	img1 = cv::imread("D:\\work\\helloOpenCV\\chap09\\1.bmp");
	img2 = cv::imread("2.bmp");
	img3 = cv::imread("3.bmp");
	img4 = cv::imread("4.jpg");

	resetBigImage(imglt, imgrt, imglb, imgrb, img1, img2, img3, img4);

	cv::imshow("stage", background);

	cv::Mat* params[9];

	params[0] = &background;
	params[1] = &imglt;
	params[2] = &imglb;
	params[3] = &imgrt;
	params[4] = &imgrb;
	params[5] = &img1;
	params[6] = &img2;
	params[7] = &img3;
	params[8] = &img4;

	cv::setMouseCallback("stage", on_mouse, &params);

	cv::waitKey(0);
	return 0;
}
*/
/*
// Exercise 9-2
cv::Rect box;
bool drawing_box = false;

void resetBigImage(
	cv::Mat& imglt, cv::Mat& imgrt, cv::Mat& imglb, cv::Mat& imgrb,
	cv::Mat& img1, cv::Mat& img2, cv::Mat& img3, cv::Mat& img4);

void draw_box(cv::Mat& img, cv::Rect& box)
{
	cv::rectangle(img, box.tl(), box.br(), cv::Scalar(0, 0, 255));
}

void on_mouse(int EVENT, int x, int y, int flags, void* params)
{
	cv::Scalar color(0, 0, 255);
	cv::String text;

	cv::Mat imglt, imgrt, imglb, imgrb;
	cv::Mat img1, img2, img3, img4;
	cv::Mat* dst = NULL;
	cv::Mat* background, *frontend;

	bool xOK = false; 
	bool yOK = false;
	bool samePic = false;

	int showX = 0;
	int showY = 0;

	char buf[64] = { 0 };

	background = *((cv::Mat * *)params);
	imglt = *(*((cv::Mat * *)params + 1));
	imgrt = *(*((cv::Mat * *)params + 2));
	imglb = *(*((cv::Mat * *)params + 3));
	imgrb = *(*((cv::Mat * *)params + 4));
	img1 = *(*((cv::Mat * *)params + 5));
	img2 = *(*((cv::Mat * *)params + 6));
	img3 = *(*((cv::Mat * *)params + 7));
	img4 = *(*((cv::Mat * *)params + 8));
	frontend = *((cv::Mat * *)params + 9);

	cv::Point p(x, y);
	cv::Point ori;

	if ((x < 300 && box.x < 300) || (x > 299 && box.x > 299)) {
		xOK = true;
	}

	if ((y < 300 && box.y < 300) || (y > 299 && box.y > 299)) {
		yOK = true;
	}

	if (xOK && yOK) {
		samePic = true;
	}

	switch (EVENT) {
	case cv::EVENT_MOUSEMOVE:
		if (drawing_box && samePic) {
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
		if (samePic) {
			draw_box(*background, box);
		}
		break;
	}
}

void resetBigImage(
	cv::Mat& imglt, cv::Mat& imgrt, cv::Mat& imglb, cv::Mat& imgrb,
	cv::Mat& img1, cv::Mat& img2, cv::Mat& img3, cv::Mat& img4)
{
	img1.rowRange(0, 300).colRange(0, 300).copyTo(imglt);
	img2.rowRange(0, 300).colRange(0, 300).copyTo(imgrt);
	img3.rowRange(0, 300).colRange(0, 300).copyTo(imglb);
	img4.rowRange(0, 300).colRange(0, 300).copyTo(imgrb);
}

int main()
{
	cv::Mat img1, img2, img3, img4;
	cv::Mat background, imglt, imgrt, imglb, imgrb;
	cv::Mat frontend;

	background = cv::Mat::zeros(600, 600, CV_8UC3);
	imglt = background.rowRange(0, 300).colRange(0, 300);
	imgrt = background.rowRange(300, 600).colRange(0, 300);
	imglb = background.rowRange(0, 300).colRange(300, 600);
	imgrb = background.rowRange(300, 600).colRange(300, 600);

	cv::namedWindow("stage");

	img1 = cv::imread("D:\\work\\helloOpenCV\\chap09\\1.bmp");
	img2 = cv::imread("2.bmp");
	img3 = cv::imread("3.bmp");
	img4 = cv::imread("4.jpg");

	resetBigImage(imglt, imgrt, imglb, imgrb, img1, img2, img3, img4);

	cv::imshow("stage", background);

	cv::Mat* params[10];

	params[0] = &background;
	params[1] = &imglt;
	params[2] = &imglb;
	params[3] = &imgrt;
	params[4] = &imgrb;
	params[5] = &img1;
	params[6] = &img2;
	params[7] = &img3;
	params[8] = &img4;
	params[9] = &frontend;

	cv::setMouseCallback("stage", on_mouse, &params);

	for (;;) {
		
		if (drawing_box) {
			resetBigImage(imglt, imgrt, imglb, imgrb, img1, img2, img3, img4);
			draw_box(background, box);
		}

		cv::imshow("stage", background);

		if (cv::waitKey(15) == 27) {
			break;
		}
	}
	
	return 0;
}
*/
/*
// Exercise 9-3
bool g_bMag = false;
int g_trackPos = 0;

cv::Point g_ptCur; // 鼠标所在位置
cv::Size g_sizeMagImg(100, 100);
cv::Mat g_image;
cv::Mat g_imgMag;

void onBtnMagnification(int state, void* params)
{
	std::cout << "state: " << state << std::endl;
	g_bMag = !g_bMag;

	if (g_imgMag.empty()) {
		g_imgMag = cv::Mat::zeros(100, 100, CV_8UC3);
	}

	if (g_bMag) {
		cv::namedWindow("amplitude", cv::WINDOW_GUI_NORMAL|cv::WINDOW_AUTOSIZE);
		cv::imshow("amplitude", g_imgMag);
	}
	else {
		cv::destroyWindow("amplitude");
	}
}

///////////////////////////////////////
// scale -- 放大倍数(1 -- 4)
// imageSize   图像尺寸           500*500
// outputSize  放大器小窗口的尺寸 100x100
//
// 以scale 反推中间图尺寸
// 1 --> 100*100
// 2 --> 50*50
// 3 --> 33*33
// 4 --> 25*25
//
//  ------------
// |           |
// |           |
// |           |
// |           |       1x
//  ------------   ------------\
//                              \
//  _______                      \
// |      |                       \
// |      |                        \     ------------
// |      |            2x           \   |           |
//  -------  ----------------------->   |           |
//                                  /   |           |
//  _____                          /    |           |
// |    |                         /      ------------
// |    |              3x        /
//  -----     ------------------/
//                             /
//  ___                4x     /
// |  |       ---------------/
//  ---
//
// 中间图：指由黑色背景及未放大的图像合成的一个中间图像
// 由于当前位置可能在边角附近，
// 需要确定从图像的哪里开始，
// 复制多大一块到中间图
///////////////////////////////////////
void makeUpMagnification(cv::Mat &image, cv::Point &pt, int scale, 
	cv::Size &outputSize, cv::Mat &outputImg)
{
	int xStart = 0;
	int yStart = 0;
	int width = 0;
	int height = 0;
	int xStartInTemp = 0;
	int yStartInTemp = 0;

	cv::Mat tempImg; // 中间图
	cv::Size tempSize;
	//
	if (scale < 1 || scale > 4) {
		return;
	}

	// 以scale 反推中间图尺寸
	tempSize.width = outputSize.width / scale;
	tempSize.height = outputSize.height / scale;

	// 生成黑背景中间图
	tempImg = cv::Mat::zeros(tempSize, CV_8UC3);

	// 以当前点位置反推应拷贝的图像尺寸及拷贝到中间图的位置
	// 正常，不考虑边角超出范围时
	xStart = pt.x - tempSize.width / 2;
	yStart = pt.y - tempSize.height / 2;
	width = tempSize.width;
	height = tempSize.height;
	xStartInTemp = 0;
	yStartInTemp = 0;

	// 根据边角情况修正
	if (pt.x < width/2) {
		xStart = 0;
		xStartInTemp = width / 2 - pt.x;
		width = width / 2 + pt.x;
	}
	else if (image.cols - pt.x < width/2) {
		width = width/2 + image.cols-pt.x;
	}

	if (pt.y < height / 2) {
		yStart = 0;
		yStartInTemp = height / 2 - pt.y;
		height = height / 2 + pt.y;
	}
	else if (image.rows - pt.y < height / 2) {
		height = height / 2 + image.rows - pt.y;
	}

	image.rowRange(yStart, yStart + height).colRange(xStart, xStart + width).copyTo(tempImg.rowRange(yStartInTemp, yStartInTemp + height).colRange(xStartInTemp, xStartInTemp + width));
	cv::resize(tempImg, tempImg, outputSize);
	tempImg.copyTo(outputImg);
}

void on_mouse(int EVENT, int x, int y, int flags, void* params)
{
	switch (EVENT) {
	case cv::EVENT_MOUSEMOVE:
		g_ptCur.x = x;
		g_ptCur.y = y;

		if (g_bMag) {
			// 根据放大倍数，以当前点为中心，显示图像
			makeUpMagnification(g_image, g_ptCur, g_trackPos, g_sizeMagImg, g_imgMag);
			cv::imshow("amplitude", g_imgMag);
		}

		break;
	}
}

int main()
{
	cv::Size size500(500, 500);
	
	cv::namedWindow("Exercise9-3", cv::WINDOW_KEEPRATIO);

	cv::createButton("magnification", onBtnMagnification, NULL, cv::QT_PUSH_BUTTON);
	cv::createTrackbar("amp", "", &g_trackPos, 3);

	cv::setMouseCallback("Exercise9-3", on_mouse);

	g_image = cv::imread("931.jpg");
	//cv::resize(image, image, cv::Size(500, 500));

	cv::imshow("Exercise9-3", g_image);
	cv::resizeWindow("Exercise9-3", size500);

	for (;;) {
		if (cv::waitKey(15) == 27) {
			break;
		}
	}
	
	return 0;
}
*/
/*
// Exercise 9-4
// 参考Exercise 8-6
bool g_bEditing = false;
int g_startX = 0;
int g_curX = 0;
int g_curY = 0;
int g_cancel = 0;
cv::Size g_char_size;
cv::Scalar g_colorBlack(0, 0, 0);
cv::Scalar g_colorWhite(255, 255, 255);
cv::String g_winName = "Exercise9-4";
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
	curColor = abs(old_char - 0x30) % 10;
	curScalar = bColorful ? g_vecColors[curColor] : g_colorWhite;
	cvstr.at(0) = old_char;
	cv::putText(img, cvstr, cvpOrigin, cv::FONT_HERSHEY_SIMPLEX, 1, curScalar);
	cv::imshow(window, img);
}

void cb_mouse(int event, int x, int y, int flags, void* param)
{
	cv::Mat& image = *(cv::Mat*)param;

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

void cb_on_cancel(int state, void* params) {
	//
	std::cout << "cb_on_cancel state:" << state << std::endl;
}

void onCancel(int pos, void* params) {
	std::cout << "on_cancel begin." << std::endl;
	cv::Mat& img = *(cv::Mat*)params;
	cv::Point cvpOrigin;

	if (g_bEditing) {
		g_bEditing = false;

		cvpOrigin.x = g_curX;
		cvpOrigin.y = g_curY;
		earse(img, cvpOrigin, g_char_size, g_colorBlack);

		cv::imshow(g_winName, img);
	}
	std::cout << "on_cancel end." << std::endl;
}

int main(int argc, char** argv) {
	cv::String cvsWindow(g_winName);

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

	cv::setMouseCallback(g_winName, cb_mouse, (void*)& img);

	cv::createButton("cancel", onCancel, &img);// need QT support
	//cv::createTrackbar("cancel", g_winName, &g_cancel, 1, onCancel, &img);

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

			// 颜色控制

			// 其他键，忽略

			//if (input_char >= 0x30 && input_char <= 0x39) {
			if (input_char >= 0x30 && input_char <= 0x7A) {
				curColor = (input_char - 0x30) % 10;

				// 擦除光标
				cvpOrigin.x = g_curX;
				cvpOrigin.y = g_curY;
				earse(img, cvpOrigin, char_size, g_colorBlack);

				cvstr.at(0) = input_char;
				text.at<uchar>(curRow, curCol) = input_char;
				cv::putText(img, cvstr, cvpOrigin, cv::FONT_HERSHEY_SIMPLEX, 1, g_vecColors[curColor]);
				cv::imshow(cvsWindow, img);

				if (g_curX < COLS * char_size.width) {
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
				cv::imshow(g_winName, img);
			}
			std::cout << "main show cursor." << std::endl;
			iCount++;
		}
		else {
			input_char = cv::waitKeyEx(50);
		}

	}

	cv::destroyWindow(g_winName);

	return 0;
}
*/
// Exercise 9-5
cv::String g_winName = "Exercise9-5";
cv::Mat g_img;
int rotx = 55;
int roty = 45;

// 是否旋转
bool g_bLeft = false;
bool g_bRight = false;
bool g_bUp = false;
bool g_bDown = false;

void RenderBackgroundTexture(cv::Mat& img)
{
	GLuint texture_ID;

	float w = img.cols;
	float h = img.rows;

	glGenTextures(1, &texture_ID);
	glBindTexture(GL_TEXTURE_2D, texture_ID);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glBindTexture(GL_TEXTURE_2D, texture_ID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, img.data);

	const GLfloat bgTextureVertices[] = { 0, 0, w, 0, 0, h, w, h };
	const GLfloat bgTextureCoords[] = { 1, 0, 1, 1, 0, 0, 0, 1 };
	const GLfloat proj[] = { 0, -2.f / w, 0, 0, -2.f / h, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1 };
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(proj);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture_ID);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	glVertexPointer(2, GL_FLOAT, 0, bgTextureVertices);
	glTexCoordPointer(2, GL_FLOAT, 0, bgTextureCoords);

	glColor4f(1, 1, 1, 1);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisable(GL_TEXTURE_2D);
	glDeleteTextures(1, &texture_ID);
}

void on_opengl2(void* param)
{
	RenderBackgroundTexture(g_img);

	cv::ogl::Texture2D* pTex = static_cast<cv::ogl::Texture2D*>(param);
	if (pTex->empty()) {
		return;
	}
	
	//glMatrixMode(GL_MODELVIEW);
	glMatrixMode(GL_PROJECTION);
	//glMatrixMode(GL_TEXTURE);
	glLoadIdentity();

	glTranslated(0.0, 0.0, 1.0);

	glRotatef(rotx, 1, 0, 0);
	glRotatef(roty, 0, 1, 0);
	glRotatef(0, 0, 0, 1);

	static const int coords[6][4][3] = {
		{ { +1, -1, -1 }, { -1, -1, -1 }, { -1, +1, -1 }, { +1, +1, -1 } },
		{ { +1, +1, -1 }, { -1, +1, -1 }, { -1, +1, +1 }, { +1, +1, +1 } },
		{ { +1, -1, +1 }, { +1, -1, -1 }, { +1, +1, -1 }, { +1, +1, +1 } },
		{ { -1, -1, -1 }, { -1, -1, +1 }, { -1, +1, +1 }, { -1, +1, -1 } },
		{ { +1, -1, +1 }, { -1, -1, +1 }, { -1, -1, -1 }, { +1, -1, -1 } },
		{ { -1, -1, +1 }, { +1, -1, +1 }, { +1, +1, +1 }, { -1, +1, +1 } }
	};

	glEnable(GL_TEXTURE_2D);
	pTex->bind();
	
	for (int i = 0; i < 6; ++i) {
		glColor3ub(i * 20, 100 + i * 10, i * 42);
		glBegin(GL_QUADS);
		for (int j = 0; j < 4; ++j) {
			glVertex3d(0.2 * coords[i][j][0], 0.2 * coords[i][j][1], 0.2 * coords[i][j][2]);
		}
		glEnd();
	}
	
}

void onChange(int pos, void* param)
{
	cv::updateWindow(g_winName);
}

void onBtnLeft(int pos, void* params)
{
	g_bLeft = !g_bLeft;
}

void onBtnRight(int pos, void* params)
{
	g_bRight = !g_bRight;
}

void onBtnUp(int pos, void* params)
{
	g_bUp = !g_bUp;
}

void onBtnDown(int pos, void* params)
{
	g_bDown = !g_bDown;
}

int main()
{
	cv::ogl::Texture2D tex;

	cv::namedWindow(g_winName, cv::WINDOW_OPENGL);
	cv::setOpenGlDrawCallback(g_winName, on_opengl2, &tex);

	cv::createTrackbar("rotx", g_winName, &rotx, 360, onChange);
	cv::createTrackbar("roty", g_winName, &roty, 360, onChange);

	cv::createButton("left", onBtnLeft);// need QT support
	cv::createButton("right", onBtnRight);// need QT support
	cv::createButton("up", onBtnUp);// need QT support
	cv::createButton("down", onBtnDown);// need QT support

	g_img = cv::imread("951.jpg");
	tex.copyFrom(g_img);
	
	cv::updateWindow(g_winName);

	for (;;) {
		if (g_bLeft) {
			rotx--;
			if (rotx < 0) {
				rotx = 360;
				cv::updateWindow(g_winName);
			}
		}
		else if (g_bRight) {
			rotx++;
			if (rotx > 360) {
				rotx = 0;
				cv::updateWindow(g_winName);
			}
		}
		else if (g_bUp) {
			roty++;
			if (roty > 360) {
				roty = 0;
				cv::updateWindow(g_winName);
			}
		}
		else if (g_bDown) {
			roty--;
			if (roty < 0) {
				roty = 360;
				cv::updateWindow(g_winName);
			}
		}

		if (cv::waitKey(15) == 27) {
			break;
		}
	}

	cv::setOpenGlDrawCallback(g_winName, 0, 0);
	cv::destroyWindow(g_winName);

	return 0;
}
