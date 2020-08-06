#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <opencv.hpp>
#include <vector>
#include <cmath>
#include <complex>
#include <time.h>

using namespace std;
using namespace cv;

typedef unsigned char BYTE;
typedef vector<BYTE> iVec;
typedef vector<iVec>vVec;
typedef vector<double> dVec;
typedef vector<dVec>ddVec;
typedef vector<int>intVec;
typedef complex<double> dcomplex;
typedef vector<dcomplex> dcVec;
typedef vector<vector<dcomplex>> dc2Vec;

#define PI 3.14159265354
#define eps 1e-6

vVec readfile(const char* _FileName, int height,int width)
{
	/*int height, width;
	if (a == 0) { height = height0; width = width0; }
	if (a == 1) { height = height1; width = width1; }
	if (a == 2) { height = height2; width = width2; }
	if (a == 3) { height = height3; width = width3; }*/
	FILE* fp = NULL;
	vVec out;
	fp = fopen(_FileName, "rb");
	if (fp == NULL)
	{
		printf("Can not read the file \r\n");
	}
	else
	{
		printf("read success!\r\n");
	}
	BYTE* dat = (BYTE*)malloc(height * width * sizeof(BYTE));
	for (int i = 0; i < height; i++)
	{
		iVec v1;
		for (int j = 0; j < width; j++)
		{
			fread(dat, 1, 1, fp);
			v1.push_back(*dat);
		}
		out.push_back(v1);
	}
	return out;
	fclose(fp);
}
Mat Vec2Mat(vVec array, int a = 0)
{
	int row = array.size();
	int col = array[0].size();
	int car;
	if (a == 0)
	{
		car = CV_8UC1;
	}
	if (a == 1)
	{
		car = CV_8UC3;
	}
	Mat img(row, col, car);
	uchar* ptmp = NULL;
	for (int i = 0; i < row; ++i)
	{
		ptmp = img.ptr<uchar>(i);

		for (int j = 0; j < col; ++j)
		{
			ptmp[j] = array[i][j];
		}
	}
	return img;
}
vVec Mat2Vec(Mat in)
{
	int row = in.rows;
	int col = in.cols;
	vector<vector<BYTE>> out;
	for (int i = 0; i < row; ++i)
	{
		vector<BYTE>v1;
		for (int j = 0; j < col; ++j)
		{
			double* pi = in.ptr<double>(i, j);
			v1.push_back(int(*pi * 255));
		}
		out.push_back(v1);
	}
	return out;
}
void showimg(vVec in, const char* _WindowName, int a = 0)
{
	Mat dst = Vec2Mat(in, a);
	namedWindow(_WindowName, WINDOW_AUTOSIZE);
	imshow(_WindowName, dst);
	imwrite(_WindowName + string(".png"), dst);
}
void Histogram(vVec in, const char* Name, vector<int> &hist)
{
	for (int i = 0; i < 256; i++) { hist.push_back(0); }

	for (int i = 0; i < in.size(); i++)
	{
		for (int j = 0; j < in[0].size(); j++)
		{
			int pixel = in[i][j];
			hist[pixel]++;
		}
	}

	Mat histmap(400, 612, CV_8UC3, Scalar(255, 255, 255));
	line(histmap, Point(50, 50), Point(50, 350), Scalar(0, 0, 0), 2);
	line(histmap, Point(50, 350), Point(562, 350), Scalar(0, 0, 0), 2);
	putText(histmap, "0", Point(40, 380), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 2, false);
	putText(histmap, "255", Point(550, 380), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 2, false);
	putText(histmap, "Pixel", Point(280, 380), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 2, false);
	int max = *max_element(hist.begin(), hist.end());
	char s[10];
	sprintf(s, "%d", max);
	for (int i = 0; i < hist.size(); i++)
	{
		int value = hist[i];
		int px = 52 + 2 * i;
		int py = value * 300 / max;
		line(histmap, Point(px, 350 - py), Point(px, 350), Scalar(0, 0, 255), 1);
		putText(histmap, s, Point(10, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1, false);
	}
	imshow(Name+string("_hist"), histmap);
	imwrite(Name+string("_hist.png"), histmap);
}
void Rayleigh(intVec hist)
{
	vector<int>::iterator max;
	max = max_element(hist.begin(), hist.end());
	int pos = distance(hist.begin(), max);
	printf("max element %d at %d\r\n", *max,pos);
	bool findA = true;
	int a = 0;
	while (findA)
	{
		if (hist[a] == 0)
		{
			a++;
		}
		else
		{
			printf("a = %d\r\n", a);
			findA = false;
		}
	}
	int b = ((pos - a) * (pos - a)) * 2;
	printf("b = %d\r\n", b);
	double u = a + sqrt(PI * b / 4);
	double sigma = (b * (4 - PI) / 4);
	printf("u = %.3f, \t sigma = %.3f \r\n", u, sigma);
}
void Gaussian(intVec hist)
{
	vector<int>::iterator max;
	max = max_element(hist.begin(), hist.end());
	int pos = distance(hist.begin(), max);
	printf("max element %d at %d\r\n", *max, pos);
	bool find_begin = true;
	bool find_end = true;
	int count = 1,begin,end;
	int sigma = *max*0.7;
	int find_sigma = 0,sigma_pos;
	while (true)
	{
		if (hist[find_sigma] > sigma)
		{
			if ((hist[find_sigma] - sigma) > (sigma - hist[find_sigma-1]))
			{
				sigma_pos = find_sigma - 1;
			}
			else
			{
				sigma_pos = find_sigma;
			}
			break;
		}
		find_sigma++;
	}
	sigma = pos - sigma_pos;
	printf("u = %d ,\t singma = %d\r\n", pos,sigma);
}
void Uniform(intVec hist)
{
	bool find_begin = true;
	bool find_end = true;
	int count = 0, begin, end;
	while (find_end)
	{
		if (hist[count] > 0 && find_begin)
		{
			begin = count;
			find_begin = false;
		}
		if (!find_begin && hist[count] == 0)
		{
			end = count - 1;
			find_end = false;
		}
		count++;
	}
	printf("a: %d \t b: %d\r\n", begin, end);
	int u = (begin + end) / 2;
	double sigma = pow((end - begin), 2) / 12;
	printf("u: %d\t sigma: %.3f\r\n", u, sigma);
}
vVec addnoise(vVec in)
{
	Mat src = Vec2Mat(in);
	imshow("sorce", src);
	imwrite("2_a_src.png", src);
	Mat noise = Mat(src.size(), CV_8S);
	randn(noise, 0, 20);
	for (int i = 0; i < in.size(); i++)
	{
		for (int j = 0; j < in[0].size(); j++)
		{
			int a = noise.at<char>(i, j);
			int src = in[i][j];
			if ((src + a) > 255)
			{
				in[i][j] = 255;
			}
			else if ((src + a) < 0)
			{
				in[i][j] = 0;
			}
			else
			{
				in[i][j] += a;
			}
		}
	}
	src = Vec2Mat(in);
	imshow("2.a", src);
	imwrite("2.a.png", src);
	return in;
}
Mat makenoise(Size size)
{
	Mat noise = Mat(size, CV_32F);
	randn(noise, 0, 20);
	return noise;
}
Mat add_pic(vVec in, int k)
{
	Mat src = Vec2Mat(in);
	Mat output;
	Mat mix = Mat(src.size(),CV_32F,Scalar::all(0)) ;
	int count= k;
	while (count != 0)
	{
		Mat add = makenoise(src.size());
		mix = mix + add;
		count--;
	}
	Mat ddd;
	ddd = mix / k;
	for (int i = 0; i < in.size(); i++)
	{
		for (int j = 0; j < in[0].size(); j++)
		{
			int a = ddd.at<float>(i, j);
			src.at<BYTE>(i, j) += a;
		}
	}
	return src;
}
vVec mirroredpadding(vVec in, int d)
{
	vVec out;
	int h0 = in.size() - 1;
	int w0 = in[0].size() - 1;
	int h = in.size() + 2 * d;
	int w = in[0].size() + 2 * d;
	for (int i = 0; i < h; i++)
	{
		iVec v1;
		for (int j = 0; j < w; j++)
		{
			int x = i - d, y = j - d;
			if (x < 0) { x = abs(x) - 1; };
			if (y < 0) { y = abs(y) - 1; };
			if (x > h0) { x = in.size() - abs(h0 - x); };
			if (y > w0) { y = in[0].size() - abs(w0 - y); };
			v1.push_back(in[x][y]);
		}
		out.push_back(v1);
	}
	return out;
}
void mediumf(vVec in, int size,const char* name)
{
	vVec out;
	int p = int(size / 2);
	vVec src = mirroredpadding(in,p);
	for (int i = 0+p; i < in.size()-p; i++)
	{
		iVec v1;
		for (int j = 0+p; j < in[0].size()-p; j++)
		{
			iVec vm;
			for (int u = -p; u <= p; u++)
			{
				for (int v = -p; v <= p; v++)
				{
					vm.push_back(in[i + u][j + v]);
				}
			}
			sort(vm.begin(), vm.end());
			v1.push_back(vm[(size*size)/2+1]);
		}
		out.push_back(v1);
	}
	Mat res = Vec2Mat(out);
	imshow(name+ string("medium"), res);
	imwrite(name + string("_medium.png"), res);
}
void meanf(vVec in, int size, const char* name)
{
	vVec out;
	int p = int(size / 2);
	vVec src = mirroredpadding(in, p);
	for (int i = 0 + p; i < in.size() - p; i++)
	{
		iVec v1;
		for (int j = 0 + p; j < in[0].size() - p; j++)
		{
			int sum = 0;
			for (int u = -p; u <= p; u++)
			{
				for (int v = -p; v <= p; v++)
				{
					sum += in[i + u][j + v];
				}
			}
			v1.push_back(sum/size/size);
		}
		out.push_back(v1);
	}
	Mat res = Vec2Mat(out);
	imshow(name + string("mean"), res);
	imwrite(name + string("_mean.png"), res);
}
dcomplex format(dcomplex& c)
{
	if (fabs(c.real()) < eps) { c.real(0); }
	if (fabs(c.imag()) < eps) { c.imag(0); }
	return c;
}
double FInf(double a) 
{
	if (a == 0) { a = 1e-6; }
	return a;
}
dc2Vec makeH(double x_m, double y_m, double T,int row,int col)
{
	double x = x_m;
	double y = y_m;
	dc2Vec out;
	for (int i = -row/2; i < row/2; i++)
	{
		dcVec v1;
		dcomplex put;
		for (int j = -col/2; j < col/2 ; j++)
		{
			double sq = ((double)i * x + (double)j * y);
			if (sq != 0)
			{
				double h = T * sin(sq * PI) / (sq * PI);
				double real = (cos(sq * PI)) * h;
				double imag = -sin(sq * PI) * h;
				put = { real,imag };
				put = format(put);
				v1.push_back(put);
			}
			else
			{
				put = { T,0 };
				v1.push_back(put);
			}
		}
		out.push_back(v1);
	}
	
	return out;
}

Mat cvDFT(vVec in)
{
	Mat img = Vec2Mat(in);
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);
	dft(complexImg, complexImg);
	split(complexImg, planes);
	magnitude(planes[0], planes[1], planes[0]);//位置相乘
	Mat mag = planes[0];
	mag += Scalar::all(1);
	log(mag, mag);
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
	Mat _magI = mag.clone();
	normalize(_magI, _magI, 0, 1, NORM_MINMAX);
	//imshow("before rearrange ", _magI);

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	// rearrange the quadrants of Fourier image
	// so that the origin is at the image center
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	normalize(mag, mag, 0, 1, NORM_MINMAX);
	//imshow("FFT", mag);
	
	return complexImg;
}
Mat cvDFT(Mat img, string Name)
{
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);
	dft(complexImg, complexImg);
	split(complexImg, planes);
	magnitude(planes[0], planes[1], planes[0]);//位置相乘
	Mat mag = planes[0];
	mag += Scalar::all(1);
	log(mag, mag);
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
	Mat _magI = mag.clone();
	normalize(_magI, _magI, 0, 1, NORM_MINMAX);
	//imshow("before rearrange ", _magI);

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	// rearrange the quadrants of Fourier image
	// so that the origin is at the image center
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	normalize(mag, mag, 0, 1, NORM_MINMAX);
	imshow(Name+string("_FFT"), mag);

	return mag;
}
Mat cvIDFT(Mat in, string Name)
{
	Mat res, invifft;
	
	idft(in, res);
	Mat planes[] = { Mat::zeros(in.size(), CV_32F),Mat::zeros(in.size(), CV_32F) };
	split(res, planes);
	magnitude(planes[0], planes[1], res);
	normalize(res, res, 0, 1, NORM_MINMAX);
	res.convertTo(invifft, CV_8UC1,255);
	//normalize(invifft, invifft, 0, 255, NORM_MINMAX);
	imshow(Name, invifft);
	imwrite(Name+string(".png"), invifft);
	return invifft;
}
Mat shiftcenter(Mat in)
{
	int w = in.rows;
	int h = in.cols;
	Mat out;
	in.copyTo(out);
	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < h; j++)
		{
			int sx = 0;
			int sy = 0;
			if (i < w / 2 && j < h / 2) { sx = w / 2 + i; sy = h / 2 + j; }
			if (i >= w / 2 && j < h / 2) { sx = i - w / 2; sy = h / 2 + j; }
			if (i < w / 2 && j >= h / 2) { sx = w / 2 + i; sy = j - h / 2; }
			if (i >= w / 2 && j >= h / 2) { sx = i - w / 2; sy = j - h / 2; }
			out.at<Vec2f>(i, j) = in.at<Vec2f>(sx, sy);
		}
	}
	return out;
}
void fftshift(const Mat& inputImg, Mat& outputImg)
{
	outputImg = inputImg.clone();
	int cx = outputImg.cols / 2;
	int cy = outputImg.rows / 2;
	Mat q0(outputImg, Rect(0, 0, cx, cy));
	Mat q1(outputImg, Rect(cx, 0, cx, cy));
	Mat q2(outputImg, Rect(0, cy, cx, cy));
	Mat q3(outputImg, Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

Mat WienerFilter(vVec in, double x_m, double y_m, double T,double K, const char* Name)
{
	int row = in.size();
	int col = in[0].size();
	dc2Vec H = makeH(x_m, y_m, T, row, col);
	Mat src_fft = cvDFT(in);
	Mat src_ffts;
	fftshift(src_fft,src_ffts);
	Mat res_fft;
	src_fft.copyTo(res_fft);
	
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			double H0 = abs(conj(H[i][j])* H[i][j]);
			if (H0 != 0)
			{
				dcomplex srcpe = { src_ffts.at<Vec2f>(i, j)[0],src_ffts.at<Vec2f>(i, j)[1] };
				dcomplex H1 = H0 / ((H[i][j]) * (H0 + K));
				srcpe *= H1;
				res_fft.at<Vec2f>(i, j)[0] = srcpe.real();
				res_fft.at<Vec2f>(i, j)[1] = srcpe.imag();
			}
			else
			{
				res_fft.at<Vec2f>(i, j)[0] = src_ffts.at<Vec2f>(i, j)[0];
				res_fft.at<Vec2f>(i, j)[1] = src_ffts.at<Vec2f>(i, j)[1];
			}
		}
	}
	Mat res_ffts;
	fftshift(res_fft,res_ffts);
	Mat res = cvIDFT(res_ffts,Name+string("_wiener_filter"));
	return res;
}
ddVec BLPF(int row,int col,int d0,int order)
{
	ddVec out;
	for (int i = -row/2 ; i < row/2; i++)
	{
		dVec v1;
		for (int j = -col/2; j < col/2; j++)
		{
			double H = 1 / (1 + (pow((i * i + j * j), order) / (pow(d0, 2 * order))));
			v1.push_back(H);
		}
		out.push_back(v1);
	}
	return out;
}
Mat InverseFilter(vVec in, double x_m, double y_m, double T, int D0,const char* Name)
{
	int row = in.size();
	int col = in[0].size();
	dc2Vec H = makeH(x_m, y_m, T, row, col);
	ddVec H_b = BLPF(row, col,D0, 10);
	Mat src_fft = cvDFT(in);
	Mat src_ffts;
	fftshift(src_fft, src_ffts);
	Mat res_fft;
	src_fft.copyTo(res_fft);

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			double H0 = abs(conj(H[i][j]) * H[i][j]);
			if (H0 != 0)
			{
				dcomplex srcpe = { src_ffts.at<Vec2f>(i, j)[0],src_ffts.at<Vec2f>(i, j)[1] };
				srcpe *= H_b[i][j]/H[i][j];
				res_fft.at<Vec2f>(i, j)[0] = srcpe.real();
				res_fft.at<Vec2f>(i, j)[1] = srcpe.imag();
			}
			else
			{
				res_fft.at<Vec2f>(i, j)[0] = src_ffts.at<Vec2f>(i, j)[0];
				res_fft.at<Vec2f>(i, j)[1] = src_ffts.at<Vec2f>(i, j)[1];
			}
		}
	}
	Mat res_ffts;
	fftshift(res_fft, res_ffts);
	Mat res = cvIDFT(res_ffts, Name+string("_Inverse_filter"));
	return res;
}
void pros1a(vVec in)
{
	imshow("input", Vec2Mat(in));
	imwrite("1_a_src.png", Vec2Mat(in));
	vector<int> hist;
	Histogram(in,"1.a",hist);
	Rayleigh(hist);
}
void pros1b(vVec in)
{
	imshow("input", Vec2Mat(in));
	imwrite("1_b_src.png", Vec2Mat(in));
	vector<int> hist;
	Histogram(in, "1.b", hist);
	int max = *max_element(hist.begin(), hist.end());
	printf("max: %d\r\n",max);
	Gaussian(hist);
}
void pros1c(vVec src,vVec cut)
{
	imshow("src", Vec2Mat(src));
	imwrite("1_c_src.png", Vec2Mat(src));
	imshow("src_cut", Vec2Mat(cut));
	imwrite("1_c_cut.png", Vec2Mat(cut));
	vector<int> hist1,hist2;
	Histogram(src, "1_c_src", hist1);
	int max = *max_element(hist1.begin(), hist1.end());
	printf("max: %d\r\n", max);
	Histogram(cut, "1_c_cut", hist2);
	max = *max_element(hist2.begin(), hist2.end());
	printf("max: %d\r\n", max);
	Uniform(hist2);
}
void pros2a(vVec in)
{
	vVec res = addnoise(in);
}
void pros2b(vVec in)
{
	Mat add7 = add_pic(in, 7);
	imshow("add7", add7);
	imwrite("add7.png", add7);
	Mat add25 = add_pic(in,25);
	imshow("add25", add25);
	imwrite("add25.png", add25);
	Mat add75 = add_pic(in, 75);
	imshow("add75", add75);
	imwrite("add75.png", add75);
}
void pros2c(vVec in)
{
	Mat src = Vec2Mat(in);
	vVec noise = addnoise(in);
	mediumf(noise, 3,"2_c_3");
	mediumf(noise, 5, "2_c_5");
	meanf(noise, 3, "2_c_3");
	meanf(noise, 5, "2_c_5");

}
void pros2d(vVec in)
{
	Mat src = Vec2Mat(in);
	imshow("sorce", src);
	mediumf(in, 3, "3");
	mediumf(in, 5, "5");
	meanf(in, 3, "3");
	meanf(in, 5, "5");

}
void pros3a(vVec in1,vVec in2)
{
	Mat src_blur = Vec2Mat(in1);
	Mat src_gau = Vec2Mat(in2);
	Mat src_blur_dft = cvDFT(src_blur,"blur_src");
	Mat res_blur_inv = InverseFilter(in1, -0.05, 0.05, 1,50,"blur");
	Mat res_blur_wie = WienerFilter(in1, -0.05, 0.05, 1, 0.01, "blur");
	Mat src_gau_dft = cvDFT(src_gau, "gau_src");
	Mat res_gau_inv = InverseFilter(in2, -0.05, 0.05, 1,12, "gau");
	Mat res_gau_wie = WienerFilter(in2, -0.05, 0.05, 1, 0.06, "gau");
}

int main()
{
	vVec p1 = readfile("noise512_a.raw", 512, 512);
	vVec p2 = readfile("noise512_b.raw", 512, 512);
	vVec p3 = readfile("hats_768x512.raw", 512, 768);
	vVec p4 = readfile("hats_768x512c.raw", 38, 72);
	vVec lena = readfile("lena512.raw", 512, 512);
	vVec lenasp = readfile("lena512_salt&pepper.raw", 512, 512);
	vVec blur = readfile("plant256_blur.raw", 256, 256);
	vVec gau = readfile("plant256_blur_gau.raw", 256, 256);
	
	char num1, num2, num3, num4;
	while (true)
	{
		double START, END;
		printf("1	--		1.	Identifying noise distribution.\r\n");
		printf("2	--		2.	Denoising.\r\n");
		printf("3	--		3.	Deblur.\r\n");
		printf("Select function: ");
		scanf(" %c", &num1);
		switch (num1)
		{
		case'1':
			printf("a	--		hist_noise512_a.\r\n");
			printf("b	--		hist_noise512_b.\r\n");
			printf("c	--		hist_hats768*512.\r\n");
			printf("Select function: ");
			scanf(" %c", &num2);
			switch (num2)
			{
			case'1':
				pros1a(p1);
				break;
			case'2':
				pros1b(p2);
				break;
			case'3':
				pros1c(p3,p4);
				//pros1c(p5);
				break;
			}
			break;
		case'2':
			printf("a	--		Add Gaussian noise.\r\n");
			printf("b	--		Gaussian.\r\n");
			printf("c	--		Noise.\r\n");
			printf("d	--		salt&pepper.\r\n");
			printf("Select function: ");
			scanf(" %c", &num3);
			switch (num3)
			{
			case'1':
				pros2a(lena);
				break;
			case'2':
				pros2b(lena);
				break;
			case'3':
				pros2c(lena);
				break;
			case'4':
				pros2d(lenasp);
				break;
			}
			break;
		case'3':
			printf("a	--		wienerfilter.\r\n");
			printf("Select function: ");
			scanf(" %c", &num4);
			switch (num4)
			{
			case'1':
				pros3a(blur,gau);
				break;
			}
			break;
		case'0':
			exit(0);
		}
		fflush(stdin);
		waitKey(0);
		destroyAllWindows();
	}
}