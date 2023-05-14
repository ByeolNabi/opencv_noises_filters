#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h> // 삼각함수

#define PI 3.141592

using namespace cv;
using namespace std;

int filter_size = 10;

void setRandomSeed() {
	srand(time(NULL));
}

namespace noise {
	Mat Gaussian(Mat input_img, int _mean, int _std) {
		Mat output_img;
		input_img.copyTo(output_img);

		// 가우시안 분포를 따르는 랜덤값을 만들기 위한 매개변수
		double mean = _mean, std = _std;
		Mat Gaussian_noise(input_img.size(), CV_16SC3); //가우시안 값 저장을 위한 행렬
		Mat Gaussian_noise_display(input_img.size(), CV_8UC3, Scalar::all(127)); //가우시안 노이즈를 표현하기 위한 행렬(중간값으로 초기화)
		cv::randn(Gaussian_noise, mean, std); // mean은 +x방향 이동값,  std가 클수록 노이즈가 심해짐
		Gaussian_noise_display += Gaussian_noise;	//노이즈 표현을 위한 연산
		imshow("Gaussian_noise", Gaussian_noise_display);
		//cout << Gaussian_noise_display(Rect(10,10,10,10));

		output_img += Gaussian_noise;
		return output_img;
	}
	Mat SaltPapper(Mat input_img, double _noise_div) {
		Mat output_img;
		input_img.copyTo(output_img);

		double mean = 0, std = 10, noise_div = _noise_div / 2;
		Mat randn_num(1, 1, CV_32F);
		randn(randn_num, mean, std); // 노이즈의 갯수를 정규분포로 결정

		// 솔트 페퍼가 잘 안 보여서 128화면에 솔트페퍼를 뿌리기로 했다.
		Mat SaltPapper_noise_display(input_img.size(), CV_8UC3, Scalar::all(128));
		Mat SaltPapper_noise(input_img.size(), CV_16SC3);
		// 해상도의 중간값(정규분포그래프의 중심을 우리가 원하는 곳으로 옮기기 위해);
		int middle = (SaltPapper_noise.rows * SaltPapper_noise.cols) / 2;
		// 뭔가 노이즈의 갯수에 규칙을 주고 싶어서 이런 공식을 사용했다.
			// 정규표준분포는 |x|<3이면 0.9974퍼를 표현한다. 따라서 middle/3 값을 곱하면 픽셀의 전체 갯수를 적당히 대응시킬수 있을 것이다.
			/* 정규분포의 x값에 middle을 더하면 전체 픽셀의 절반정도의 숫자가 뽑힌다(정규분포그래프의 중점이 middle이 된다).
				그러나 절반의 노이즈는 너무 많기 때문에 전체의 1/8픽셀만큼 노이즈를 주기로 결정했다..*/
		int noises = randn_num.at<double>(0, 0) * middle / 3 + middle / noise_div;

		for (int i = 0; i < noises; ++i) {	// 노이즈 갯수만큼 반복(중복 허용)
			// 랜덤 x,y 만들기
			int x = rand() % SaltPapper_noise.cols;
			int y = rand() % SaltPapper_noise.rows;
			int noiseColor = rand() % 2 ? -255 : 255;	// 솔트페퍼 노이즈를 덧셈으로 구현하기 위해서 -255를 사용한다.
			SaltPapper_noise.at<cv::Vec3s>(y, x)[0] = noiseColor;
			SaltPapper_noise.at<cv::Vec3s>(y, x)[1] = noiseColor;
			SaltPapper_noise.at<cv::Vec3s>(y, x)[2] = noiseColor;
		}
		SaltPapper_noise_display += SaltPapper_noise;	// 노이즈 시각화
		imshow("SaltPapper_noise", SaltPapper_noise_display);
		//cout << SaltPapper_noise_display(Rect(10,10,10,10));

		output_img += SaltPapper_noise;
		return output_img;
	}
	Mat Periodic(Mat input_img, double _period, double _range) {
		Mat output_img;
		input_img.copyTo(output_img);

		Mat Periodic_noise(output_img.size(), CV_8SC3);	// 노이즈 저장을 위한 행렬
		// 밝고 어두움의 주기 n번 반복
		double period = _period;
		// 진폭의 크기
		double range = _range;

		for (int row = 0; row < Periodic_noise.rows; ++row) {
			uchar* pointer_row = Periodic_noise.ptr<uchar>(row);
			for (int col = 0; col < Periodic_noise.cols; ++col) {
				double bias = sin(col * (2 * PI * period) / Periodic_noise.rows);
				pointer_row[col * 3 + 0] += bias * range;
				pointer_row[col * 3 + 1] += bias * range;
				pointer_row[col * 3 + 2] += bias * range;
			}
		}
		// CV_8SC3은 imshow가 잘 되네? 16U부터는 일반적인 255로는 표현이 잘 안 되던데. 16에서 8로 변환할 수 있다면 saltNoise 표현이 쉬울 것 같다.
		imshow("Periodic_noise", Periodic_noise);
		//cout << Periodic_noise(Rect(10, 10, 100, 10));

		output_img += Periodic_noise;
		return output_img;
	}
}

void histogram(string name ,Mat inputImg) {
	Mat src = inputImg;
	vector<Mat> bgr_planes;
	split(src, bgr_planes);
	int histSize = 256;
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };
	bool uniform = true, accumulate = false;
	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	int hist_w = 768, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	for (int i = 0; i < 255; i++){
		line(histImage, Point(bin_w * (i), hist_h),
			Point(bin_w * (i), hist_h - b_hist.at<float>(i)),
			Scalar(255, 0, 0));
		line(histImage, Point(bin_w * (i), hist_h),
			Point(bin_w * (i), hist_h - g_hist.at<float>(i)),
			Scalar(0, 255, 0));
		line(histImage, Point(bin_w * (i), hist_h),
			Point(bin_w * (i), hist_h - r_hist.at<float>(i)),
			Scalar(0, 0, 255));
	}

	imshow(name, histImage);
}

int main() {
	setRandomSeed();
	Mat src = imread("./images/Lenna.jpg");
	//Mat src = imread("./images/Lenna.jpg");
	imshow("src image", src);

	// 노이즈 첨가
	Mat Gaussian_noise_img = noise::Gaussian(src, 0, 40);
	imshow("Gaussian image", Gaussian_noise_img);

	Mat SaltPapper_noise_img = noise::SaltPapper(src, 8);
	imshow("SaltPapper image", SaltPapper_noise_img);

	Mat Periodic_noise_img = noise::Periodic(src, 10, 30);
	imshow("Periodic image", Periodic_noise_img);

	//================================//
	// 가우시안 노이즈 억제
	Mat Gaussian_aF;	// 평균
	blur(Gaussian_noise_img, Gaussian_aF, Size(5, 5));
	imshow("Gaussian a filter", Gaussian_aF);
	Mat Gaussian_gF;	// 가우시안
	GaussianBlur(Gaussian_noise_img, Gaussian_gF, Size(5, 5), 0, 0);
	imshow("Gaussian g filter", Gaussian_gF);
	Mat Gaussian_mF;	// 중간값
	medianBlur(Gaussian_noise_img, Gaussian_mF, 5);
	imshow("Gaussian m filter", Gaussian_mF);

	// 솔트페퍼 노이즈 억제
	Mat SaltPapper_aF;
	blur(SaltPapper_noise_img, SaltPapper_aF, Size(5, 5));
	imshow("SaltPapper a filter", SaltPapper_aF);
	Mat SaltPapper_gF;
	GaussianBlur(SaltPapper_noise_img, SaltPapper_gF, Size(5, 5), 0, 0);
	imshow("SaltPapper g filter", SaltPapper_gF);
	Mat SaltPapper_mF;
	medianBlur(SaltPapper_noise_img, SaltPapper_mF, 5);
	imshow("SaltPapper m filter", SaltPapper_mF);

	// 주기 노이즈 억제
	Mat Periodic_aF;
	blur(Periodic_noise_img, Periodic_aF, Size(5, 5));
	imshow("Periodic a filter", Periodic_aF);
	Mat Periodic_gF;
	GaussianBlur(Periodic_noise_img, Periodic_gF, Size(5, 5), 0, 0);
	imshow("Periodic g filter", Periodic_gF);
	Mat Periodic_mF;
	medianBlur(Periodic_noise_img, Periodic_mF, 5);
	imshow("Periodic m filter", Periodic_mF);

	//// 원본과 차이
	//// 뺄셈
	//Mat Gaussian_sub_img(src.size(), CV_8UC3);
	//Mat SaltPapper_sub_img(src.size(), CV_8UC3);
	//Mat Periodic_sub_img(src.size(), CV_8UC3);

	//Gaussian_sub_img = abs(Gaussian_noise_img - src);
	//SaltPapper_sub_img = abs(SaltPapper_noise_img - src);
	//Periodic_sub_img = abs(Periodic_noise_img - src);

	//imshow("Gaussian sub image", Gaussian_sub_img);
	//imshow("SaltPapper sub image", SaltPapper_sub_img);
	//imshow("Periodic sub image", Periodic_sub_img);

	//histogram("src histo", src);

	waitKey(0);
	return 0;
}
