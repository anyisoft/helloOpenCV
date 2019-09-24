// chap07.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
/*
// Exercise 7-1
int main()
{
	cv::RNG rng = cv::theRNG();

	float f1 = rng.uniform(0.f, 1.f);
	float f2 = rng.uniform(0.f, 1.f);
	float f3 = rng.uniform(0.f, 1.f);

	std::cout << "f1: " << f1 << " f2: " << f2 << " f3: " << f3 << std::endl;

	double d1 = rng.gaussian(1.0);
	double d2 = rng.gaussian(1.0);
	double d3 = rng.gaussian(1.0);

	std::cout << "d1: " << d1 << " d2: " << d2 << " d3: " << d3 << std::endl;

	uchar uc1 = rng.uniform(0, 255);
	uchar uc2 = rng.uniform(0, 255);
	uchar uc3 = rng.uniform(0, 255);

	std::cout << "uc1: " << (int)uc1 << " uc2: " << (int)uc2 << " uc3: " << (int)uc3 << std::endl;

	cv::waitKey();
}
*/
/*
// Exercise 7-2
int main()
{
	cv::RNG rng = cv::theRNG();

	std::cout << "UNIFORM 20" << std::endl;
	cv::Mat img = cv::Mat::zeros(1, 20, CV_32FC1);
	rng.fill(img, cv::RNG::UNIFORM, 0.0, 1.0);
	//std::cout.setf(std::ios::left);
	//std::cout.width(20);
	//std::cout.fill(' ');
	for (int i = 0; i < 20; i++) {
		if (i != 0 && 0 == i % 5) {
			std::cout << std::endl;
		}
		std::cout.width(12);
		std::cout << img.at<float>(0, i);
	}
	std::cout << std::endl;

	std::cout << "NORMAL 20" << std::endl;
	cv::Mat ufo20 = cv::Mat::zeros(1, 20, CV_32FC1);
	rng.fill(ufo20, cv::RNG::NORMAL, 0.0, 1.0);
	for (int i = 0; i < 20; i++) {
		if (i != 0 && 0 == i % 5) {
			std::cout << std::endl;
		}
		std::cout.width(12);
		std::cout << ufo20.at<float>(0, i);
	}
	std::cout << std::endl;

	std::cout << "COLOR NORMAL 20" << std::endl;
	cv::Mat uco20 = cv::Mat::zeros(1, 20, CV_8UC3);
	rng.fill(uco20, cv::RNG::UNIFORM, 0, 255);
	for (int i = 0; i < 20; i++) {
		if (i != 0 && 0 == i % 5) {
			std::cout << std::endl;
		}
		std::cout << "(";
		std::cout.width(4);
		std::cout << (int)uco20.at<cv::Vec3b>(0, i)[0];
		std::cout << ",";
		std::cout.width(4);
		std::cout << (int)uco20.at<cv::Vec3b>(0, i)[1];
		std::cout << ",";
		std::cout.width(4);
		std::cout << (int)uco20.at<cv::Vec3b>(0, i)[2];
		std::cout << ") ";
	}
	std::cout << std::endl;

	cv::waitKey();
}
*/
/*
// 参考博客：http://blog.sina.com.cn/s/blog_802a94a20102vq0f.html
int main()
{
	float Coordinates[20] = { 
		1.5, 2.3, 3.0, 1.7, 1.2, 
		2.9, 2.1, 2.2, 3.1, 3.1, 
		1.3, 2.7, 2.0, 1.7, 1.0, 
		2.0, 0.5, 0.6, 1.0, 0.9 
	};
	cv::Mat originalMat(10, 2, CV_32FC1, Coordinates);
	std::cout << "original mat:" << std::endl;
	std::cout << originalMat << std::endl;
	cv::PCA pca(originalMat, cv::noArray(), cv::PCA::DATA_AS_ROW, 2);
	std::cout << "eigenvalues=" << pca.eigenvalues << std::endl;
	std::cout << "eigenvectors=" << pca.eigenvectors << std::endl;
	cv::Mat dst = pca.project(originalMat);
	std::cout << "projected mat:" << std::endl;
	std::cout << dst << std::endl;
	cv::Mat backProjected = pca.backProject(dst);
	std::cout << "back projected mat;" << std::endl;
	std::cout << backProjected << std::endl;

	cv::waitKey();

	return 0;
}
*/
/*
// Exercise 7-3
int main()
{
	cv::RNG rng = cv::theRNG();

	std::cout << "COLOR NORMAL 100" << std::endl;
	cv::Mat uco100 = cv::Mat::zeros(1, 100, CV_8UC3);
	std::vector<uchar> vecMean;
	std::vector<uchar> vecStdev;

	vecMean.push_back(64);
	vecMean.push_back(192);
	vecMean.push_back(128);

	vecStdev.push_back(10);
	vecStdev.push_back(10);
	vecStdev.push_back(2);

	rng.fill(uco100, cv::RNG::NORMAL, vecMean, vecStdev);
	for (int i = 0; i < 100; i++) {
		if (i != 0 && 0 == i % 5) {
			std::cout << std::endl;
		}
		std::cout << "(";
		std::cout.width(4);
		std::cout << (int)uco100.at<cv::Vec3b>(0, i)[0];
		std::cout << ",";
		std::cout.width(4);
		std::cout << (int)uco100.at<cv::Vec3b>(0, i)[1];
		std::cout << ",";
		std::cout.width(4);
		std::cout << (int)uco100.at<cv::Vec3b>(0, i)[2];
		std::cout << ") ";
	}
	std::cout << std::endl;

	std::cout << "COLOR NORMAL 100 C1" << std::endl;

	// PCA 只接受单通道数据
	cv::Mat uco100c1 = cv::Mat::zeros(100, 3, CV_8UC1);

	rng.fill(uco100c1.col(0), cv::RNG::NORMAL, vecMean[0], vecStdev[0]);
	rng.fill(uco100c1.col(1), cv::RNG::NORMAL, vecMean[1], vecStdev[1]);
	rng.fill(uco100c1.col(2), cv::RNG::NORMAL, vecMean[2], vecStdev[2]);
	for (int i = 0; i < 100; i++) {
		if (i != 0 && 0 == i % 5) {
			std::cout << std::endl;
		}
		std::cout << "(";
		std::cout.width(4);
		std::cout << (int)uco100c1.at<uchar>(i, 0);
		std::cout << ",";
		std::cout.width(4);
		std::cout << (int)uco100c1.at<uchar>(i, 1);
		std::cout << ",";
		std::cout.width(4);
		std::cout << (int)uco100c1.at<uchar>(i, 2);
		std::cout << ") ";
	}
	std::cout << std::endl;

	cv::Scalar uco100c1_mean0 = cv::mean(uco100c1.col(0));
	std::cout << "uco100c1 mean_0: " << uco100c1_mean0 << std::endl;

	cv::Scalar uco100c1_mean1 = cv::mean(uco100c1.col(1));
	std::cout << "uco100c1 mean_1: " << uco100c1_mean1 << std::endl;

	cv::Scalar uco100c1_mean2 = cv::mean(uco100c1.col(2));
	std::cout << "uco100c1 mean_2: " << uco100c1_mean2 << std::endl;

	//std::cout << "COLOR NORMAL 100 C1 opencv output: " << std::endl;
	//std::cout << uco100c1 << std::endl;

	//cv::PCA pca(uco100c1, cv::Mat(), cv::PCA::DATA_AS_ROW);
	//cv::PCA pca(uco100c1, vecMean, cv::PCA::DATA_AS_ROW, 2);
	cv::PCA pca(uco100c1, cv::Mat(), cv::PCA::DATA_AS_ROW, 2);

	cv::Mat pca_res = pca.project(uco100c1);
	//std::cout << "project result: " << std::endl;
	//std::cout << pca_res << std::endl;

	cv::Mat pca_bres = pca.backProject(pca_res);
	// 注意这里的pca_bres 和pca_res 的成员是32位浮点数，
	// 否则调试时直接查看data 地址的数据会有困惑
	std::cout << "back project result: " << std::endl;
	std::cout << pca_bres << std::endl;

	cv::Scalar pca_bres_mean0 = cv::mean(pca_bres.col(0));
	std::cout << "back project mean_0: " << pca_bres_mean0 << std::endl;

	cv::Scalar pca_bres_mean1 = cv::mean(pca_bres.col(1));
	std::cout << "back project mean_1: " << pca_bres_mean1 << std::endl;

	cv::Scalar pca_bres_mean2 = cv::mean(pca_bres.col(2));
	std::cout << "back project mean_2: " << pca_bres_mean2 << std::endl;

	cv::waitKey();
}
*/
// Exercise 7-3
int main()
{
	cv::SVD svd;

	cv::Mat mW;
	cv::Mat mU;
	cv::Mat mVt;
	cv::Mat mA = cv::Mat::zeros(3, 2, CV_32FC1);
	mA.at<float>(0, 0) = 1;
	mA.at<float>(0, 1) = 1;
	mA.at<float>(1, 0) = 0;
	mA.at<float>(1, 1) = 1;
	mA.at<float>(2, 0) = -1;
	mA.at<float>(2, 1) = 1;

	cv::SVD::compute(mA, mW, mU, mVt);// , cv::SVD::NO_UV);

	std::cout << "mA: " << std::endl << mA << std::endl << std::endl;
	std::cout << "mW: " << std::endl << mW << std::endl << std::endl;
	std::cout << "mU: " << std::endl << mU << std::endl << std::endl;
	std::cout << "mVt: " << std::endl << mVt << std::endl << std::endl;

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
