﻿// chap04.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>
//#include <opencv2\videoio.hpp>
//#include <opencv2\video.hpp>

cv::Scalar g_colorBlack(0, 0, 0);
cv::Scalar g_colorWhite(255, 255, 255);
std::vector<cv::Scalar> g_vecColors;


// 获取字符尺寸
cv::Size OutputCharWidthHeight(const cv::String &cvstr)
{
	int baseLine = 0;
	
	cv::Size char_size = cv::getTextSize(cvstr, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);
	std::cout << cvstr << "(w, h) is (" << char_size.width << ", " << char_size.height << ")" << std::endl;

	return char_size;
}

// 擦除背景
// 另：使用putText 写空格的方法进行背景处理是无效的
void earse(cv::Mat &img, cv::Point &cvpOrigin, const cv::Size &char_size, 
	const cv::Scalar &colorBackGround)
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

void restore(cv::Mat &img, cv::Mat &text, int curRow, int curCol, 
	const cv::Size &char_size, const cv::Scalar& colorBackGround,
	const cv::String &window, bool bColorful
)
{
	int curColor = 0;
	cv::Point cvpOrigin;
	cv::String cvstr(" ");
	cv::Scalar curScalar;

	cvpOrigin.x = curCol * char_size.width;
	cvpOrigin.y = (curRow + 1) * char_size.height;
	earse(img, cvpOrigin, char_size, colorBackGround);
	// 重写当前字符
	uchar old_char = text.at<uchar>(curRow, curCol);
	if (isdigit(old_char)) {
		curColor = old_char - 0x30;
		curScalar = bColorful ? g_vecColors[curColor] : g_colorWhite;
		cvstr.at(0) = old_char;
		cv::putText(img, cvstr, cvpOrigin, cv::FONT_HERSHEY_SIMPLEX, 1, curScalar);
		cv::imshow(window, img);
	}
}

void colorful(cv::Mat &img, cv::Mat &text, const cv::Size& char_size,
	const cv::String& window, bool bColorful)
{
	int curColor = 0;
	uchar curChar;
	cv::Point cvpOrigin;
	cv::String cvstr(" ");
	cv::Scalar curScalar;

	// 重画文字
	for (int i = 0; i < text.rows; i++) {
		for (int j = 0; j < text.cols; j++) {
			curChar = text.at<uchar>(i, j);
			if (isdigit(curChar ) ) {
				curColor = curChar - 0x30;
				curScalar = bColorful ? g_vecColors[curColor] : g_colorWhite;
				cvstr.at(0) = curChar;
				cvpOrigin.x = j * char_size.width;
				cvpOrigin.y = (i + 1) * char_size.height;
				cv::putText(img, cvstr, cvpOrigin, cv::FONT_HERSHEY_SIMPLEX, 1, curScalar);
			}
		}
	}
	
	cv::imshow(window, img);
}

int main()
{
    std::cout << "Chapter 4 examples!\n";
	/*
	cv::Mat m = cv::Mat::eye(10, 10, CV_32FC1);
	std::cout << "Element(3, 3) is " << m.at<float>(3, 3) << std::endl;
	*/
	/*
	cv::Mat m = cv::Mat::eye(10, 10, CV_32FC2);
	printf("Element (3, 3) is (%f, %f)\n",
		m.at<cv::Vec2f>(3, 3)[0],
		m.at<cv::Vec2f>(3, 3)[1]);
	*/
	/*
	cv::Mat m = cv::Mat::eye(10, 10, cv::DataType<cv::Complexf>::type);
	printf("Element (3, 3) is %f + i%f)\n",
		m.at<cv::Complexf>(3, 3).re,
		m.at<cv::Complexf>(3, 3).im);
	*/
	/*
	int sz[3]{ 4, 4, 4 };
	cv::Mat m(3, sz, CV_32FC3);
	cv::randu(m, -1.0f, 1.0f);

	float max = 0.0f;
	float len2 = 0.0f;
	int line = 0;
	cv::MatConstIterator_<cv::Vec3f> it = m.begin<cv::Vec3f>();
	while (it != m.end<cv::Vec3f>()) {
		len2 = (*it)[0]*(*it)[0] + (*it)[1]*(*it)[1] + (*it)[2]*(*it)[2];
		if (len2 > max) {
			max = len2;
		}
		std::cout << "No" << line << ": len2: " << len2 << std::endl;
		it++;
		line++;
	}
	*/
	/*
	const int n_mat_size = 5;
	const int n_mat_sz[] = { n_mat_size, n_mat_size, n_mat_size };
	cv::Mat n_mat0(3, n_mat_size, CV_32FC1);
	cv::Mat n_mat1(3, n_mat_size, CV_32FC1);

	cv::RNG rng;
	rng.fill(n_mat0, cv::RNG::UNIFORM, 0.f, 1.f);
	rng.fill(n_mat1, cv::RNG::UNIFORM, 0.f, 1.f);

	const cv::Mat* arrays[] = { &n_mat0, &n_mat1, 0 };
	cv::Mat my_planes[2];
	cv::NAryMatIterator it(arrays, my_planes);

	float s = 0.f;
	int n = 0;
	for (int p = 0; p < it.nplanes; p++, ++it) {
		s += cv::sum(it.planes[0])[0];
		s += cv::sum(it.planes[1])[0];
		std::cout << "No" << n << ": s: " << s << std::endl;
		n++;
	}
	*/
	/*
	// 打印一个稀疏矩阵中的所有非0 元素
	int size[] = { 10, 10 };
	cv::SparseMat sm(2, size, CV_32F);

	for (int i = 0; i < 10; i++) {
		int idx[2];
		idx[0] = size[0] * rand();
		idx[1] = size[1] * rand();
		sm.ref<float>(idx) += 1.0f;
	}

	cv::SparseMatConstIterator_<float> it = sm.begin<float>();
	cv::SparseMatConstIterator_<float> it_end = sm.end<float>();

	for (; it != it_end; ++it) {
		const cv::SparseMat::Node* node = it.node();
		printf("(%3d, %3d) %f\n", node->idx[0], node->idx[1], *it);
	}
	*/
	
	// Exercise4-1
	cv::String cvsWindow("Exercise4-1");

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

	while (true) {

		// 当前光标处背景擦除
		if (0 == (iCount/10)%2) {
			cvpOrigin.x = curCol * char_size.width;
			cvpOrigin.y = (curRow + 1) * char_size.height;
			earse(img, cvpOrigin, char_size, g_colorBlack);
			cv::imshow(cvsWindow, img);
		}
		
		input_char = cv::waitKeyEx(50);
		//std::cout << "input_char: 0x"  << std::hex << input_char << std::endl;

		if (27 == input_char) {
			break;
		}

		// 数字键，在当前光标处输出数字，然后光标前进

		// 方向键，移动光标

		// 回车

		// 退格

		// 颜色控制

		// 其他键，忽略

		if (input_char >= 0x30 && input_char <= 0x39) {
			curColor = input_char - 0x30;
			
			// 擦除光标
			cvpOrigin.x = curCol * char_size.width;
			cvpOrigin.y = (curRow + 1) * char_size.height;
			earse(img, cvpOrigin, char_size, g_colorBlack);

			cvstr.at(0) = input_char;
			text.at<uchar>(curRow, curCol) = input_char;
			cv::putText(img, cvstr, cvpOrigin, cv::FONT_HERSHEY_SIMPLEX, 1, g_vecColors[curColor]);
			cv::imshow(cvsWindow, img);

			curCol++;
			if (curCol >= COLS) {
				curCol = 0;
				curRow++;
				if (curRow >= ROWS) {
					curRow = 0;
				}
			}
		}
		else if (0x08 == input_char) { // 退格
			// 
			restore(img, text, curRow, curCol, char_size, g_colorBlack, cvsWindow, bColorful);

			curCol--;
			if (curCol < 0) {
				curCol = COLS-1;
				curRow--;
				if (curRow < 0) {
					curRow = ROWS-1;
				}
			}
			text.at<uchar>(curRow, curCol) = 0;
		}
		else if (0x250000 == input_char) {
			// left
			// 擦除光标
			// 重写当前字符
			restore(img, text, curRow, curCol, char_size, g_colorBlack, cvsWindow, bColorful);

			curCol--;
			if (curCol < 0) {
				curCol = COLS-1;
				curRow--;
				if (curRow < 0) {
					curRow = ROWS-1;
				}
			}
		}
		else if (0x260000 == input_char) {
			// top
			// 擦除光标
			// 重写当前字符
			restore(img, text, curRow, curCol, char_size, g_colorBlack, cvsWindow, bColorful);

			curRow--;
			if (curRow < 0) {
				curRow = ROWS-1;
			}
		}
		else if (0x270000 == input_char) {
			// right
			// 擦除光标
			// 重写当前字符
			restore(img, text, curRow, curCol, char_size, g_colorBlack, cvsWindow, bColorful);

			curCol++;
			if (curCol > COLS-1) {
				curCol = 0;
				curRow++;
				if (curRow >= ROWS) {
					curRow = 0;
				}
			}
		}
		else if (0x280000 == input_char) {
			// down
			// 擦除光标
			// 重写当前字符
			restore(img, text, curRow, curCol, char_size, g_colorBlack, cvsWindow, bColorful);

			curRow++;
			if (curRow > ROWS-1) {
				curRow = 0;
			}
		}
		else if (0xd == input_char) {
			// 回车
			// 擦除光标
			// 重写当前字符
			restore(img, text, curRow, curCol, char_size, g_colorBlack, cvsWindow, bColorful);

			curRow++;
			if (curRow > ROWS - 1) {
				curRow = 0;
			}
			curCol = 0;
		}
		else if (0x43 == input_char || 0x63 == input_char) {
			// 颜色控制
			bColorful = !bColorful;
			colorful(img, text, char_size, cvsWindow, bColorful);
		}

		// 显示光标
		if (1 == (iCount / 10) % 2) {
			cvpOrigin.x = curCol * char_size.width;
			cvpOrigin.y = (curRow + 1) * char_size.height;
			cv::putText(img, cvstrCursor, cvpOrigin, cv::FONT_HERSHEY_SIMPLEX, 1, g_colorWhite);
			cv::imshow("Exercise4-1", img);
		}
		
		iCount++;
	}
	
	cv::destroyWindow("Exercise4-1");

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
