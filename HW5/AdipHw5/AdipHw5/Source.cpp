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
typedef vector<iVec> vVec;
typedef vector<double> dVec;
typedef vector<dVec> ddVec;
typedef complex<double> dcomplex;
typedef vector<dcomplex> dcVec;
typedef vector<vector<dcomplex>> dc2Vec;

#define height3 64; 
#define width3 64; 
#define height0 128; 
#define width0 128; 
#define height1 256; 
#define width1 256; 
#define height2 512; 
#define width2 512; 
#define PI 3.14159265354
#define eps 1e-1



vVec readfile(const char* _FileName, int a)
{
	int height, width;
	if (a == 0) { height = height0; width = width0; }
	if (a == 1) { height = height1; width = width1; }
	if (a == 2) { height = height2; width = width2; }
	if (a == 3) { height = height3; width = width3; }
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
const char* strmix(const char* Name)
{
	char  c[30];
	strncpy_s(c,Name, 30);
	Name = strcat(c, ".png");
	return Name;
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
			double *pi = in.ptr<double>(i, j);
			v1.push_back(int(*pi*255));
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
	char  c[30];
	strncpy_s(c, _WindowName, 30);
	imwrite(strcat(c, ".png"), dst);
}
//設定門檻
dcomplex format(dcomplex& c)
{
	if (fabs(c.real()) < eps) { c.real(0); }
	if (fabs(c.imag()) < eps) { c.imag(0); }
	return c;
}		
double format(double& c) {
	if (fabs(c) < eps) c = 0;
	return c;
}
//double->BYTE
vVec d2v(ddVec in)
{
	vVec out;
	int row = in.size();
	int col = in[0].size();
	for (int i = 0; i < row; ++i)
	{
		iVec v1;
		for (int j = 0; j < col; ++j)
		{
			v1.push_back(int(in[i][j]));
		}
		out.push_back(v1);
	}
	return out;
}

void show_filter(ddVec in, const char* Name)
{
	int row = in.size();
	int col = in[0].size();
	double max = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (max < in[i][j])
			{
				max = in[i][j];
			}
		}
	}
	
	ddVec show;
	for (int i = 0; i < row; i++) {
		dVec v1;
		for (int j = 0; j < col; j++) {
			v1.push_back(in[i][j] / max * 255);
		}
		show.push_back(v1);
	}
	Mat showf = Vec2Mat(d2v(show));
	imshow("filter", showf);
	string c = "_filter.png";
	c = Name + c;
	imwrite(c, showf);
}
vVec mNormalize(ddVec in)
{
	double max = 0;
	double min = 1000;
	for (int i = 0; i < in.size(); i++)
	{
		for (int j = 0; j < in[0].size(); j++)
		{
			if (in[i][j] > max) { max = in[i][j]; }
			if (in[i][j] < min) { min = in[i][j]; }
		}
	}
	vVec out;
	for (int i = 0; i < in.size(); i++)
	{
		iVec v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			v1.push_back(255 * (in[i][j]-min) / (max-min));
		}
		out.push_back(v1);
	}
	return out;
}
vVec shiftcenter(vVec in)
{
	int w = in.size();
	int h = in[0].size();
	vVec out;
	for (int i = 0; i < w; i++)
	{
		iVec v1;
		for (int j = 0; j < h; j++)
		{
			int sx = 0;
			int sy = 0;
			if (i < w / 2 && j < h / 2) { sx = w / 2 + i; sy = h / 2 + j; }
			if (i >= w / 2 && j < h / 2) { sx = i - w / 2; sy = h / 2 + j; }
			if (i < w / 2 && j >= h / 2) { sx = w / 2 + i; sy = j - h / 2; }
			if (i >= w / 2 && j >= h / 2) { sx = i - w / 2; sy = j - h / 2; }
			v1.push_back(in[sx][sy]);
		}
		out.push_back(v1);
	}
	return out;
}
ddVec shiftcenter(ddVec in)
{
	int w = in.size();
	int h = in[0].size();
	ddVec out;
	for (int i = 0; i < w; i++)
	{
		dVec v1;
		for (int j = 0; j < h; j++)
		{
			int sx = 0;
			int sy = 0;
			if (i < w / 2 && j < h / 2) { sx = w / 2 + i; sy = h / 2 + j; }
			if (i >= w / 2 && j < h / 2) { sx = i - w / 2; sy = h / 2 + j; }
			if (i < w / 2 && j >= h / 2) { sx = w / 2 + i; sy = j - h / 2; }
			if (i >= w / 2 && j >= h / 2) { sx = i - w / 2; sy = j - h / 2; }
			v1.push_back(in[sx][sy]);
		}
		out.push_back(v1);
	}
	return out;
}
/*vVec shiftcenter_1d(vVec in,const char* type)
{
	int w = in.size();
	int h = in[0].size();
	vVec out;
	for (int i = 0; i < w; i++)
	{
		iVec v1;
		for (int j = 0; j < h; j++)
		{
			int sx = 0;
			int sy = 0;
			if (type == "r")
			{
				if (j < h / 2) { sx = i, sy = j + h / 2; }
				if (j >= h / 2) { sx = i, sy = j - h / 2; }
			}
			if (type == "c")
			{
				if (i < w / 2) { sx = w / 2 + i, sy = j; }
				if (i >= w / 2) { sx = i - w / 2; sy = j; }
			}
			v1.push_back(in[sx][sy]);
		}
		out.push_back(v1);
	}
	return out;
}*/
//一次做2維
void mDFT(vVec in, const char* Name)
{
	int M = in.size();
	int N = in[0].size();
	ddVec out1;
	double fx = (-2 * PI) / M;
	double fy = (-2 * PI) / N;
	for (int u = 0; u < M; u++)
	{
		dVec v1;
		for (int v = 0; v < N; v++)
		{
			dcomplex t{ 0, 0 };
			for (int x = 0; x < M; x++)
			{
				for (int y = 0; y < M; y++)
				{
					double px = u * x * fx;
					double py = v * y * fy;
					double real = in[x][y] * cos(px + py);
					double imag = in[x][y] * sin(px + py);
					t += {real, imag};
				}
			}
			t = format(t);
			double m = abs(t);
			v1.push_back(format(m));
		}
		out1.push_back(v1);
	}
	vVec res = mNormalize(out1);
	res = shiftcenter(res);
	showimg(res, Name);
}
//做兩次一維
vector<vector<dcomplex>> mDFT2(vVec in, const char* Name)
{
	int M = in.size();
	int N = in[0].size();
	vector<vector<dcomplex>> out1;
	vector<vector<dcomplex>> data;
	ddVec out2,outp;
	double fx = (-2 * PI) / M;
	double fy = (-2 * PI) / N;
	for (int x = 0; x < M; x++)
	{
		vector<dcomplex> v1;
		for (int v = 0; v < N; v++)
		{
			dcomplex t;
			for (int y = 0; y < M; y++)
			{
				double py = v * y * fy;
				double real = cos(py) * in[x][y];
				double imag = sin(py) * in[x][y];
				t += {real, imag};
			}
			v1.push_back(t);
		}
		out1.push_back(v1);
	}

	for (int u = 0; u < M; u++)
	{
		vector<dcomplex> vc1;
		dVec v1,v2;
		for (int v = 0; v < N; v++)
		{
			dcomplex t;
			for (int x = 0; x < M; x++)
			{
				double px = u * x * fx;
				double real = cos(px) * out1[x][v].real()-sin(px) * out1[x][v].imag();
				double imag = sin(px) * out1[x][v].real()+cos(px) * out1[x][v].imag();
				t += {real, -imag};
			}
			t = format(t);
			double m = log(abs(t)+1);
			double p = atanf(t.imag()/t.real());
			v1.push_back(m);
			v2.push_back(p);
			vc1.push_back(t);
		}
		outp.push_back(v2);
		out2.push_back(v1);
		data.push_back(vc1);
	}
	vVec res = shiftcenter(mNormalize(out2));
	vVec res2 = shiftcenter(mNormalize(outp));
	showimg(res, Name);
	char  c[30];
	strncpy_s(c, Name, 30);
	showimg(res2, strcat(c, "_p"));
	return data;
}
void mIDFT2(dc2Vec in, const char* Name)
{
	int M = in.size();
	int N = in[0].size();
	vector<vector<dcomplex>> cos1,sin1;
	ddVec out2;
	double fx = (-2 * PI) / M;
	double fy = (-2 * PI) / N;
	for (int x = 0; x < M; x++)
	{
		vector<dcomplex> vc,vs;
		for (int v = 0; v < N; v++)
		{
			dcomplex t,t1;
			for (int y = 0; y < M; y++)
			{
				double py = v * y * fy;
				double real = cos(py) * in[x][y].real();  //cos*A
				double imag = cos(py) * in[x][y].imag();  //cos *B
				double real1 = sin(py) * in[x][y].real(); //sin * A
				double imag1 = sin(py) * in[x][y].imag();  //sin * B
				t += {real, imag};
				t1 += {real1, imag1};
			}
			vc.push_back(t);
			vs.push_back(t1);
		}
		cos1.push_back(vc);
		sin1.push_back(vs);
	}
	for (int u = 0; u < M; u++)
	{
		dVec v1;
		for (int v = 0; v < N; v++)
		{
			dcomplex t;
			for (int x = 0; x < M; x++)
			{
				double px = u * x * fx;
				double real = cos(px) * cos1[x][v].real() - sin(px) * sin1[x][v].real() + sin(px) * cos1[x][v].imag() + cos(px) * sin1[x][v].imag();
				double imag = -sin(px) * cos1[x][v].real() -cos(px) * sin1[x][v].real()+ cos(px) * cos1[x][v].imag() - sin(px) * sin1[x][v].imag();
				t += {real, imag};
			}
			t = format(t);
			double m = abs(t);
			v1.push_back(m/M/N);
		}
		out2.push_back(v1);
	}
	vVec res = (mNormalize(out2));
	showimg(res, Name);
}
Mat cvDFT(vVec in ,const char* Name)
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
	imshow(Name, mag);
	Mat res1;
	mag.convertTo(res1, CV_8U, 255);
	string c = ".png";
	c = Name + c;
	imwrite(c, res1);
	//vVec res = Mat2Vec(mag);
	return complexImg;
}
void phase(Mat in,const char* Name)
{
	

	int row = in.rows;
	int col = in.cols;
	ddVec p;
	for (int i = 0; i < row; i++)
	{
		dVec v1;
		for (int j = 0; j < col; j++)
		{
			v1.push_back( atan(in.at<Vec2f>(i, j)[1] / in.at<Vec2f>(i, j)[0]));
		}
		p.push_back(v1);
	}

	char  c[30];
	strncpy_s(c, Name, 30);
	showimg(shiftcenter(mNormalize(p)), strcat(c, "_phase"));
}
Mat medthod1(Mat input, vVec mark)
{
	Mat out;
	input.copyTo(out);
	int row = input.rows;
	int col = input.cols;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (mark[i][j] == 255)
			{
				out.at<Vec2f>(i, j)[0] *= 0.3;
				out.at<Vec2f>(i, j)[1] *= 0.3;
			}
		
		}
	}
	return out;
}
Mat med(vVec in, const char* Name,vVec mask)
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
	complexImg = medthod1(complexImg, mask);
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
	imshow(Name, mag);
	Mat res1;
	mag.convertTo(res1, CV_8U,255);
	string c = ".png";
	c = Name + c;
	imwrite(c, res1);
	//vVec res = Mat2Vec(mag);
	return complexImg;
}
Mat medthod2(Mat input, Mat mark)
{
	Mat out;
	input.copyTo(out);
	int row = input.rows;
	int col = input.cols;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			out.at<Vec2f>(i, j)[0] += (mark.at<Vec2f>(i, j)[0])*0.01;
			out.at<Vec2f>(i, j)[1] += (mark.at<Vec2f>(i, j)[1])*0.01;
		}
	}
	
	return out;
}
Mat cvIDFT(Mat in, const char* Name)
{
	Mat res, invifft;
	Mat planes1[] = { Mat::zeros(in.size(), CV_32F),Mat::zeros(in.size(), CV_32F) };
	split(in, planes1);
	magnitude(planes1[0], planes1[1], planes1[0]);//位置相乘
	Mat mag = planes1[0];
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
	//imshow(Name + string("_mag"), mag);
	Mat res1;
	mag.convertTo(res1, CV_8U, 255);
	imwrite(Name + string("_mag.png"), res1);
	/*split(in, planes1);
	magnitude(planes1[0], planes1[1], res);
	normalize(res, res, 0, 1, NORM_MINMAX);
	imshow(string(Name+string("_mag")), res);*/
	idft(in, res);
	Mat planes[] = { Mat::zeros(in.size(), CV_32F),Mat::zeros(in.size(), CV_32F) };
	split(res, planes);
	magnitude(planes[0], planes[1], res);
	normalize(res, res, 0, 1, NORM_MINMAX);
	imshow(Name, res);
	
	res.convertTo(invifft, CV_8U,255);
	/*normalize(invifft, invifft, 0, 255, NORM_MINMAX);
	imshow(Name, invifft);*/

	char  c[30];
	strncpy_s(c, Name, 30);
	imwrite(strcat(c, ".png"), invifft);
	return invifft;
}
Mat filter(Mat in,ddVec ft)
{
	Mat out;
	in.copyTo(out);
	int row = in.rows;
	int col = in.cols;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			out.at<Vec2f>(i, j)[0] *= ft[i][j];
			out.at<Vec2f>(i, j)[1] *= ft[i][j];
		}
	}
	return out;
}
Mat denoise(Mat in,ddVec mark)
{
	Mat out;
	in.copyTo(out);
	int row = in.rows;
	int col = in.cols;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			out.at<Vec2f>(i, j)[0] *= (1 - mark[i][j]);
			out.at<Vec2f>(i, j)[1] *= (1 - mark[i][j]);
		}
	}
	return out;
}
ddVec gau_spa(int row, int col, int d,int x,int y)
{
	int cx = row-x;
	int cy = col-y;
	ddVec out;
	double max = 0;
	for (int i = -x; i < -x+row; i++)
	{
		dVec v1;
		for (int j = -y; j < -y+col; j++)
		{
			double r = sqrt(i * i + j * j);
			double g = exp((-(r * r)) / (2 * d * d));
			v1.push_back(g);
			if (g > max)
			{ 
				max = g; 
			}
		}
		out.push_back(v1);
	}
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			out[i][j] /= max;
		}
	}
	
	return out;
}
ddVec LPF(int row,int col,int d)
{
	ddVec out;
	int cx = row / 2;
	int cy = col / 2;
	for (int  i = 0; i < row; i++)
	{
		dVec v1;
		for (int j = 0; j < col; j++)
		{
			double dis = sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy));
			if (dis >= d)
			{
				v1.push_back(0.0);
			}
			else
			{
				v1.push_back(1.0);
			}
		}
		out.push_back(v1);
	}
	return out;
}
ddVec HPF(int row, int col, int d)
{
	ddVec out;
	int cx = row / 2;
	int cy = col / 2;
	for (int i = 0; i < row; i++)
	{
		dVec v1;
		for (int j = 0; j < col; j++)
		{
			double dis = sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy));
			if (dis <= d)
			{
				v1.push_back(0);
			}
			else
			{
				v1.push_back(1);
			}
		}
		out.push_back(v1);
	}
	return out;
}
ddVec Gaussian_LPF(int row, int col, double d)
{
	ddVec out;
	double max = 0;
	for (int i = -row/2; i < row/2; i++)
	{
		dVec v1;
		for (int j = -col/2; j < col/2; j++)
		{
			double r = sqrt(i * i + j * j);
			double g = exp((-(r * r) ) / (2 * d * d));
			v1.push_back(g);
			if (g > max)
			{
				max = g;
			}
		}
		out.push_back(v1);
	}
	return out;
}
ddVec Gaussian_HPF(int row, int col, int d)
{
	ddVec out;
	ddVec gL= Gaussian_LPF(row, col, d);
	for (int i = 0; i < row; i++)
	{
		dVec v1;
		for (int j = 0; j < col ; j++)
		{
			double g = 1.0 - gL[i][j];
			v1.push_back(g);
		}
		out.push_back(v1);
	}
	return out;
}

ddVec defilter(int d)
{
	ddVec out;
	ddVec filter1 = gau_spa(512, 512, d, 256, 192);
	ddVec filter2 = gau_spa(512, 512, d, 211, 211);
	ddVec filter3 = gau_spa(512, 512, d, 192, 256);
	ddVec filter4 = gau_spa(512, 512, d, 211, 301);
	ddVec filter5 = gau_spa(512, 512, d, 256, 320);
	ddVec filter6 = gau_spa(512, 512, d, 301, 301);
	ddVec filter7 = gau_spa(512, 512, d, 320, 256);
	ddVec filter8 = gau_spa(512, 512, d, 301, 211);
	double max = 0;
	for (int i = 0; i < 512; i++)
	{
		dVec v1;
		for (int j = 0; j < 512; j++)
		{
			double m = filter1[i][j] + filter2[i][j] + filter3[i][j] + filter4[i][j] + filter5[i][j] + filter6[i][j] + filter7[i][j] + filter8[i][j];
			if (m < 0.2) { m = 0; }
			if (m > 0.8) { m = 1; }
			v1.push_back(m);
			if (m > max)
			{
				max = m;
			}
		}
		out.push_back(v1);
	}
	return out;
}

void pros1(vVec in, const char* Name, vector<vector<dcomplex>> &data)
{
	data = mDFT2(in, Name);
}
void pros2(vVec in, const char* Name)
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
	imshow(Name, mag);
	normalize(mag, mag, 0, 255, NORM_MINMAX);
	char  c[30];
	strncpy_s(c, Name, 30);
	imwrite(strcat(c, ".png"), mag);
	phase(complexImg, Name);
}
void pros3(dc2Vec in, const char* Name)
{
	mIDFT2(in, Name);
}
void pros4(vVec in)
{
	Mat res = cvDFT(in, "input");
	ddVec LPF5 = LPF(512, 512, 5);
	show_filter(LPF5,"LPF5");
	ddVec LPF20 = LPF(512, 512, 20);
	show_filter(LPF20, "LPF20");
	ddVec LPF60 = LPF(512, 512, 60);
	show_filter(LPF60, "LPF60");
	ddVec HPF5 = HPF(512, 512, 5);
	show_filter(HPF5, "HPF5");
	ddVec HPF20 = HPF(512, 512, 20);
	show_filter(HPF20, "HPF20");
	ddVec HPF60 = HPF(512, 512, 60);
	show_filter(HPF60, "HPF60");
	cvIDFT(filter(res, shiftcenter(LPF5)), "2.a.LPF5");
	cvIDFT(filter(res, shiftcenter(LPF20)), "2.a.LPF20");
	cvIDFT(filter(res, shiftcenter(LPF60)), "2.a.LPF60");
	cvIDFT(filter(res, shiftcenter(HPF5)), "2.a.HPF5");
	cvIDFT(filter(res, shiftcenter(HPF20)), "2.a.HPF20");
	cvIDFT(filter(res, shiftcenter(HPF60)), "2.a.HPF60");
}
void pros5(vVec in)
{
	Mat res = cvDFT(in, "input");
	ddVec g_LPF5 = Gaussian_LPF(512, 512, 5);
	show_filter(g_LPF5, "g_LPF5");
	ddVec g_LPF20 = Gaussian_LPF(512, 512, 20);
	show_filter(g_LPF20, "g_LPF20");
	ddVec g_LPF60 = Gaussian_LPF(512, 512, 60);
	show_filter(g_LPF60, "g_LPF60");
	ddVec g_HPF5 = Gaussian_HPF(512, 512, 5);
	show_filter(g_HPF5, "g_HPF5");
	ddVec g_HPF20 = Gaussian_HPF(512, 512, 20);
	show_filter(g_HPF20, "g_HPF20");
	ddVec g_HPF60 = Gaussian_HPF(512, 512, 60);
	show_filter(g_HPF60, "g_HPF60");
	cvIDFT(filter(res, shiftcenter(g_LPF5)), "2.b.G_LPF5");
	cvIDFT(filter(res, shiftcenter(g_LPF20)), "2.b.G_LPF20");
	cvIDFT(filter(res, shiftcenter(g_LPF60)), "2.b.G_LPF60");
	cvIDFT(filter(res, shiftcenter(g_HPF5)), "2.b.G_HPF5");
	cvIDFT(filter(res, shiftcenter(g_HPF20)), "2.b.G_HPF20");
	cvIDFT(filter(res, shiftcenter(g_HPF60)), "2.b.G_HPF60");
}
void pros6(vVec normal,vVec noise)
{
	Mat res = cvDFT(normal, "normal");
	Mat res1 = cvDFT(noise, "noise");
	ddVec filter1 = defilter(8);
	show_filter(filter1, "gau8");
	cvIDFT(denoise(res1, shiftcenter(filter1)), "2.3");
	//denoise(res1);
}
void pros7(vVec in, vVec mark)
{
	Mat res = cvDFT(in, "lena");
	Mat mark1 = Vec2Mat(mark);
	imshow("mark", mark1);
	Mat res1 = med(in, "3.1.mask_fq", shiftcenter(mark));
	cvIDFT(res1, "3.1.mask");
}
void pros8(vVec in, vVec mark)
{
	Mat lena = cvDFT(in, "lena");
	Mat mark1 = cvDFT(mark, "mark_f");
	cvIDFT(medthod2(lena,mark1), "3.2_res");
}

int main()
{
	vVec p1 = readfile("blackwhite256_rotate.raw", 1);
	vVec p2 = readfile("sine16_256.raw", 1);
	vVec p3 = readfile("fox.raw",2);
	vVec p4 = readfile("fox_noise.raw", 2);
	vVec p5 = readfile("lena256.raw", 1);
	vVec p6 = readfile("mark256.raw", 1);



	vector<vector<dcomplex>> data1;
	vector<vector<dcomplex>> data2;

	char num1, num2,num3,num4;
	while (true)
	{
		double START, END;
		printf("1	--		1. 2D-DFT.\r\n");
		printf("2	--		2. Filter in Frequency.\r\n");
		printf("3	--		3.Water in Frequency.\r\n");
		printf("Select function: ");
		scanf(" %c", &num1);
		switch (num1)
		{
		case'1':
			printf("a	--		DFT.\r\n");
			printf("b	--		OpenCV_DFT.\r\n");
			printf("c	--		IDFT.\r\n");
			printf("Select function: ");
			scanf(" %c", &num2);
			switch (num2)
			{
			case'1':
				START = clock();
				pros1(p1, "1.a.1",data1);
				END = clock();
				printf("Spend time: %.0f s\r\n", ((END - START)/CLOCKS_PER_SEC));
				START = clock();
				pros1(p2, "1.a.2",data2);
				END = clock();
				printf("Spend time: %.0f s\r\n", ((END - START)/CLOCKS_PER_SEC));
				break;
			case'2':
				START = clock();
				pros2(p1,"1.b.1");
				END = clock();
				printf("Spend time: %.0f ms\r\n",((END-START)));
				START = clock();
				pros2(p2,"1.b.2");
				END = clock();
				printf("Spend time: %.0f ms\r\n", ((END - START)));
				break;
			case'3':
				pros1(p1, "1.a.1", data1);
				START = clock();
				pros3(data1, "1.c.1");
				END = clock();
				printf("Spend time: %.0f s\r\n", ((END - START) / CLOCKS_PER_SEC));
				pros1(p2, "1.a.2", data2);
				START = clock();
				pros3(data2, "1.c.2");
				END = clock();
				printf("Spend time: %.0f s\r\n", ((END - START) / CLOCKS_PER_SEC));
				break;
			}
			break;
		case'2':
			printf("a	--		ideal.\r\n");
			printf("b	--		Gaussian.\r\n");
			printf("c	--		Noise.\r\n");
			printf("Select function: ");
			scanf(" %c", &num3);
			switch (num3)
			{
			case'1':
				pros4(p3);
				break;
			case'2':
				pros5(p3);
				break;
			case'3':
				pros6(p3,p4);
				break;
			}
			break;
		case'3':
			printf("a	--		method i.\r\n");
			printf("b	--		method ii.\r\n");
			printf("Select function: ");
			scanf(" %c", &num4);
			switch (num4)
			{
			case'1':
				pros7(p5,p6);
				break;
			case'2':
				pros8(p5,p6);
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