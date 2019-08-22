// chap03.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <math.h>
#include <time.h>
#include <opencv2/core.hpp>

/**
 * exercise 1
 * find and open the ../opencv/cxcore/include/cxtypes.h, 
 * read it and you can found many functions that convert a type to another
 * a. select a negative float number.
 * b. calculate its absolute value, approximate value, ceiling value 
 *    and floor value.
 * c. produce some random number.
 * d. make an instance of cv::Point2f, convert it to the cv::Point, 
 *    convert cvPoint to cv::Point2f.
 */
/*
int main()
{
	float f = -19.132019822;

    std::cout << "chap03 exercise 1!\n";

	std::cout << "f: " << f << std::endl; 
	std::cout << "absolute value: " << abs(f) << std::endl;
	std::cout << "approximate value: " << round(f) << std::endl;
	std::cout << "ceiling value: " << ceil(f) << std::endl;
	std::cout << "floor value: " << floor(f) << std::endl;

	srand(time(0));
	for (int i = 0; i < 10; i++) {
		std::cout << "the random number: " << rand() << std::endl;
	}

	cv::Point2f cvp2f(f, f + 33.6);
	std::cout << "cvp2f: " << cvp2f << std::endl;

	cv::Point cvp(cvp2f);
	std::cout << "cvp(cvf2f): " << cvp << std::endl;
}
*/
/**
 * exercise 2
 * 紧凑型矩阵和向量类。
 * a. 用cv::Mat33f 和cv::Vec3f 对象，对应生成一个3*3 的矩阵和3 行的向量。
 * b. 可以直接把它们两个相乘吗？如果不行，为什么不行？
 * c. 
 */
/*
int main()
{
	cv::Vec3f  cvv3f(3.14, 2.36, 3.88);
	cv::Vec3f  cvv3f_1(9.14, 7.36, 2.88);
	cv::Vec3f  cvv3f_2(4.14, 8.36, 5.88);
	
	cv::Matx33f  cvm33f(
		cvv3f[0],
		cvv3f[1],
		cvv3f[2],
		cvv3f_1[0],
		cvv3f_1[1],
		cvv3f_1[2],
		cvv3f_2[0],
		cvv3f_2[1],
		cvv3f_2[2]
	);

	std::cout << "cvv33f: " << cvv3f << std::endl;
	std::cout << "cvv33f_1: " << cvv3f_1 << std::endl;
	std::cout << "cvv33f_2: " << cvv3f_2 << std::endl;
	std::cout << "cvm33f: " << cvm33f << std::endl;
}
*/
/**
 * exercise 3
 * 紧凑型矩阵和向量模板类。
 * a. 用cv::Mat<> 和cv::Vec模板，对应生成一个3*3 的矩阵和3 行的向量。
 * b. 可以直接把它们两个相乘吗？如果不行，为什么？
 * c. 尝试类型映射，用cv::Mat<>模板把向量对象映射到3*1 的矩阵，
 *    看看会发生什么情况？
 */
 int main()
 {
	 cv::Vec3i  cvv3i(3, 2, 9);
	 cv::Vec3i  cvv3i_1(9, 7, 2);
	 cv::Vec3i  cvv3i_2(4, 8, 5);

	 /*cv::Vec<cv::Vec3f, 3> cvv33f(
		 cvv3f[0],
		 cvv3f[1],
		 cvv3f[2],
		 cvv3f_1[0],
		 cvv3f_1[1],
		 cvv3f_1[2],
		 cvv3f_2[0],
		 cvv3f_2[1],
		 cvv3f_2[2]
	 );*/
	 cv::Vec<cv::Vec3i, 3> cvv33i(cvv3i, cvv3i_1, cvv3i_2);
	 
	 
	 // std::vector can work well
	 //std::vector<cv::Vec3f> cvv33f;
	 //cvv33f.push_back(cvv3f);
	 //cvv33f.push_back(cvv3f_1);
	 //cvv33f.push_back(cvv3f_2);
	 
	 cv::Matx<int, 3, 3>  cvm33i(
		 cvv3i[0],
		 cvv3i[1],
		 cvv3i[2],
		 cvv3i_1[0],
		 cvv3i_1[1],
		 cvv3i_1[2],
		 cvv3i_2[0],
		 cvv3i_2[1],
		 cvv3i_2[2]
	 );

	 std::cout << "cvv33i: " << cvv3i << std::endl;
	 std::cout << "cvv33i_1: " << cvv3i_1 << std::endl;
	 std::cout << "cvv33i_2: " << cvv3i_2 << std::endl;
	 //std::cout << "cvv33i: " << cvv33i << std::endl;
	 std::cout << "cvm33i: " << cvm33i << std::endl;

	 cv::Matx<cv::Vec<int, 3>, 3, 1> cvm_v3_31i(
		 cvv3i[0],
		 cvv3i[1],
		 cvv3i[2],
		 cvv3i_1[0],
		 cvv3i_1[1],
		 cvv3i_1[2],
		 cvv3i_2[0],
		 cvv3i_2[1],
		 cvv3i_2[2]
	 );
	 //cvm33i.mul(cvv33i);
	 //cvm33i.mul(cvm_v3_31i);
	 //std::cout << "cvm_v3_31i: " << cvm_v3_31i << std::endl;
	 
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
