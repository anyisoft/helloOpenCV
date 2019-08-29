// chap05.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

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

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
