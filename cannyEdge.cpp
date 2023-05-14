#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

Mat src, src_gray;
Mat dst, detected_edges;
int edgeThresh = 1;
int lowThreshold = 0;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;

static void CannyThreshold(int, void*) {
	blur(src, detected_edges, Size(3, 3));
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);
	dst = Scalar::all(0);
	src.copyTo(dst, detected_edges);
	imshow("Image", src);
	imshow("Canny", dst);
}

// 가우시안 필터
Mat Gaussian_blur(Mat input);
// Sobel_xy필터를 적용시킨 결과물
Mat Sobel_xy(Mat input);
// 경사에 따른 NMS
Mat Gradiant_NMS(Mat input);

// Canny필터인데 히스테리시스를 뺀 
Mat Canny_filter(Mat input);



int main() {
	/*Mat*/ src = imread("./images/Lenna.jpg", IMREAD_GRAYSCALE);
	imshow("src", src);

	Mat dst = Canny_filter(src);
	imshow("canny_result", dst);

	src = imread("./images/Lenna.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) { return -1; }
	dst.create(src.size(), src.type());
	namedWindow("Canny", WINDOW_AUTOSIZE);
	createTrackbar("Min Threshold:", "Canny", &lowThreshold, max_lowThreshold, CannyThreshold);
	CannyThreshold(0, 0);

	waitKey(0);
	return 0;
}

/*##################################*/
Mat Gaussian_blur(Mat input) {
	Mat dst;
	input.copyTo(dst);
	GaussianBlur(input, dst, Size(5, 5), 0, 0);

	return dst;
}
Mat Sobel_xy(Mat input) {
	Mat dst;
	input.copyTo(dst);

	Mat grad;
	int scale = 1;
	int delta = 0;

	/* sobel은 기울기 근사를 구하는 필터이다.
	x방향 y방향을 각각 구해야하기 때문에 두개의 mat을 생성한다.*/
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	Sobel(dst, grad_x, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(dst, grad_y, CV_16S, 0, 1, 3, scale, delta, BORDER_DEFAULT);

	/* 엣지의 정의가 밝기가 급격하게 변하는 곳이라고 한다면,
	abs를 적용 했을 때에 국소적 최댓값만 찾으면 되기 때문에 오히려 편해진다.*/
	convertScaleAbs(grad_x, abs_grad_x);
	imshow("sobel_x", abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	imshow("sobel_y", abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	return grad;
}
Mat Gradiant_NMS(Mat input) {
	Mat NMS_x, NMS_y;
	input.copyTo(NMS_x); input.copyTo(NMS_y);
	int cols = NMS_x.cols, rows = NMS_y.rows;
	// x++ 방향 비최대억제
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols - 1; ++x) { // 국소 최대 gradiant만 저장
			if (input.at<uchar>(y, x) <= input.at<uchar>(y, x + 1)) {
				NMS_x.at<uchar>(y, x) = 0;
			}
			else {
				NMS_x.at<uchar>(y, x + 1) = 0;
			}
		}
	}
	// y++ 방향 비최대억제
	for (int x = 0; x < cols; ++x) {
		for (int y = 0; y < rows - 1; ++y) { // 국소 최대 gradiant만 저장
			if (input.at<uchar>(y, x) <= input.at<uchar>(y + 1, x)) {
				NMS_y.at<uchar>(y, x) = 0;
			}
			else {
				NMS_y.at<uchar>(y + 1, x) = 0;
			}
		}
	}
	Mat NMS = NMS_x + NMS_y;	// 그냥 더해도 되나?
	NMS *= 100;
	return NMS;
}

Mat Canny_filter(Mat input) {
	Mat dst;
	input.copyTo(dst);
	// 노이즈 억제
	dst = Gaussian_blur(dst); imshow("Gaussian", dst);
	// 기울기 합산
	dst = Sobel_xy(dst); imshow("Sobel_xy", dst);
	// 비최대 억제 (Non-maximum Suppression)
	dst = Gradiant_NMS(dst); imshow("NMS", dst);

	return dst;
}