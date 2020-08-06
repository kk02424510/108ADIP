#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <opencv.hpp>
#include <vector>
#include <time.h>
#include <cmath>

using namespace std;
using namespace cv;
typedef unsigned char BYTE;
typedef vector<BYTE> iVec;
typedef vector<iVec> vVec;
typedef vector<double> fVec;
typedef vector<fVec> mVec;

#define height1 256; 
#define width1 256; 
#define height2 512; 
#define width2 512; 
#define height3 720; 
#define width3 480; 


double Scale_4(double argu)
{
	return floor(argu * 10.0 + 0.5) / 10.0;
}
vVec readfile(const char* _FileName, int a)
{
	int height, width;
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
void showimg(vVec in, const char* _WindowName, int a = 0)
{
	Mat dst = Vec2Mat(in, a);
	namedWindow(_WindowName, WINDOW_AUTOSIZE);
	imshow(_WindowName, dst);
	char  c[30];
	strncpy_s(c, _WindowName, 30);
	imwrite(strcat(c,".png"), dst);
}
vVec zeropadding(vVec in, int d)
{
	vVec out;
	int h = in.size() + 2 * d;
	int w = in[0].size() + 2 * d;
	for (int i = 0; i < h; i++)
	{
		iVec v1;
		for (int j = 0; j < w; j++)
		{
			if (i == 0 || j == 0 || i == (h - 1) || j == (w - 1))
			{
				v1.push_back(0);
			}
			else
			{
				v1.push_back(in[i - 1][j - 1]);
			}
		}
		out.push_back(v1);
	}
	return out;
}
vVec replicatedpadding(vVec in, int d)
{
	vVec out;
	int h = in.size() + 2 * d;
	int w = in[0].size() + 2 * d;
	for (int i = 0; i < h; i++)
	{
		iVec v1;
		for (int j = 0; j < w; j++)
		{
			int x = i - 1, y = j - 1;
			if (x < 0) { x = 0; };
			if (y < 0) { y = 0; };
			if (x > in.size() - 1) { x = in.size() - 1; };
			if (y > in[0].size() - 1) { y = in[0].size() - 1; };
			v1.push_back(in[x][y]);
		}
		out.push_back(v1);
	}
	return out;
}
vVec mirroredpadding(vVec in, int d)
{
	vVec out;
	int h0 = in.size()-1;
	int w0 = in[0].size()-1;
	int h = in.size() + 2 * d;
	int w = in[0].size() + 2 * d;
	for (int i = 0; i < h; i++)
	{
		iVec v1;
		for (int j = 0; j < w; j++)
		{
			int x = i - d, y = j - d;
			if (x < 0) { x = abs(x)-1; };
			if (y < 0) { y = abs(y)-1; };
			if (x > h0) { x = in.size() - abs(h0-x); };
			if (y > w0) { y = in[0].size() - abs(w0-y); };
			v1.push_back(in[x][y]);
		}
		out.push_back(v1);
	}
	return out;
}
mVec Gaussian(int size, double sigma)
{
	int a = -size / 2;
	int b = size / 2;
	double PI =3.1415;
	double sum = 0.0;
	mVec out;
	for (int i = a; i <=b; i++)
	{
		fVec v1;
		for (int j = a; j <=b; j++)
		{
			double a = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * sigma * sigma * PI);
			v1.push_back(a);
			sum += a;
		}
		out.push_back(v1);
	}

	for (int i = 0; i < size	; i++) {
		for (int j = 0; j < size; j++) {
			out[i][j] /= sum;
		}
	}
return out;
}


vVec filter(vVec in, mVec mask, const char* _type = "m",double di = 1)
{
	vVec out;
	vVec paded;
	int ms = mask.size();
	int m = mask.size() / 2;
	if (_type == "z")
	{
		paded = zeropadding(in, m);
	}
	if (_type == "r")
	{
		paded = replicatedpadding(in, m);
	}
	if (_type == "m")
	{
		paded = mirroredpadding(in, m);
	}
	int h = paded.size();
	int w = paded[0].size();
	for (int i = m; i < h - m; i++)
	{
		iVec v1;
		for (int j = m; j < w - m; j++)
		{
			double pi = 0;
			for (int s = 0; s < ms; s++)
			{
				for (int t = 0; t < ms; t++)
				{
					pi = pi + paded[i + s - m][j + t - m] * mask[s][t];
				}
			}
			pi = pi / di;
			if (pi > 255) { pi = 255; }
			if (pi < 0) { pi = 0; }
			v1.push_back(pi);
		}
		out.push_back(v1);
	}
	return out;
}
vVec clear(vVec in1, vVec in2)
{
	for (int i = 0; i < in1.size(); i++)
	{
		iVec v1;
		for (int j = 0; j < in1[0].size(); j++)
		{
			int pi = in1[i][j] - in2[i][j];
			pi = in1[i][j] + pi;
			in2[i][j] = pi;
		}
	}
	return in2;
}

mVec normalized(mVec in)
{
	double size = in.size();
	int cen = size / 2;
	double sum = 0.0;
	double d = size*size/in[cen][cen];
	for (int i = 0; i < in.size(); i++) {
		for (int j = 0; j < in[0].size(); j++) {
			in[i][j]  = int(in[i][j]*d);
			sum += in[i][j];
		}
	}
	while (sum != 0)
	{
		double mean = sum / size / size;
		double e_sum = sum;
		sum = 0.0;
		for (int i = 0; i < in.size(); i++) {
			for (int j = 0; j < in[0].size(); j++) {
				if (in[i][j] < 0 && mean >0.5)
				{
					in[i][j] = int(in[i][j] - mean-0.5);
				}
				if (i==cen && j == cen && mean <0.5)
				{
					in[i][j] = in[i][j]- e_sum;
				}
				sum += in[i][j];
			}
		}
	}
	
	return in;
}

vVec DoG(vVec in, int size, double sigma1, double sigma2)
{
	mVec mask1 = Gaussian(size, sigma1);
	mVec mask2 = Gaussian(size, sigma2);
	mVec Dog;
	double sum = 0.0;
	for (int i = 0; i < size; i++)
	{
		fVec v1;
		for (int j = 0; j < size; j++)
		{
			double value = mask1[i][j] - mask2[i][j];
			v1.push_back(value);
			sum += value;
		}
		Dog.push_back(v1);
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			Dog[i][j] /= sum;
		}
	}
	vVec res = filter(in, Dog);
	
	
	return res;
}
mVec LoG(int size, double sigma,double &sum)
{
	int a = -size / 2;
	int b = size / 2;
	double PI = 3.1415;
	mVec out;
	double min = 0;
	for (int i = a; i <=b; i++)
	{
		fVec v1;
		for (int j = a; j <= b; j++)
		{
			int s = abs(i);
			int t = abs(j);
			double v = exp(-(s* s + t * t) / (2.0 * sigma * sigma)) / (sigma*sigma*sigma * sigma * PI) * ((s * s + t * t) / (2.0 * sigma * sigma)-1);
			v1.push_back(v);
			sum += v;
			//if (v > min) { min =v; }
		}
		out.push_back(v1);
	}
	
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			out[i][j] /= sum;
		}
	}
	out = normalized(out);
return out;
}
void Histogram(vector<vector<BYTE>>in,const char* Name)
{
	vector<int> hist;
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
	/*char str1[30] = { 0 };
	strncpy_s(str1, Name, 30);
	char str2[] = "_hist";
	strcat(str1, str2);*/
	char c[30];
	strncpy_s(c, Name, 30);
	//c = const_cast<char*>(Name);
	imshow(strcat(c,"_hist"), histmap);
	imwrite(strcat(c,".png"), histmap);
}
void mse_psnr(vector<vector<BYTE>>in1, vector<vector<BYTE>>in2)
{
	double mse, psnr;
	int sum_sq = 0;
	int w = in1.size();
	int h = in1[0].size();
	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < h; j++)
		{
			int err = in2[i][j] - in1[i][j];
			sum_sq += (err * err);
		}
	}
	mse = (double)sum_sq / (h * w);
	psnr = 10 * log10(255 * 255 / mse);

	printf("mse : %f, psnr : %f", mse, psnr);
}
void pros1(vVec in)
{
	mVec mask = { {0,-1,0}, { -1 ,4 ,-1},{0 ,-1,0} };
	mVec mask1 = { {-1,-1,-1},{-1,8,-1},{-1,-1,-1} };
	vVec out1z = filter(in, mask, "z");
	vVec out2z = filter(in, mask1, "z");
	vVec out1r = filter(in, mask, "r");
	vVec out2r = filter(in, mask1, "r");
	showimg(out1z, "1_a_1z");
	showimg(out2z, "1_a_2z");
	showimg(out1r, "1_a_1r");
	showimg(out2r, "1_a_2r");
}
void pros2(vVec in)
{
	mVec mask = { {-1,0,1}, { -2 ,0 ,2},{-1 ,0,1} };
	mVec mask1 = { {-1,-2,-1},{0,0,0},{1,2,1} };
	vVec out1 = filter(in, mask);
	vVec out2 = filter(in, mask1);
	showimg(out1, "1_b_1");
	showimg(out2, "1_b_2");
}
void pros3(vVec in1,vVec in2)
{
	double m2 = 1.0 / 25.0;
	mVec mask = { {m2,m2,m2,m2,m2},{m2,m2,m2,m2,m2},{m2,m2,m2,m2,m2},{m2,m2,m2,m2,m2},{m2,m2,m2,m2,m2} };
	vVec out1 = filter(in1, mask);
	vVec out2 = filter(in2, mask);
	showimg(out1, "2_a_1");
	showimg(out2, "2_a_2");
	Histogram(out1, "2_a_1");
	Histogram(out2, "2_a_2");
	Mat hist1, hist2;

}
void pros4(vVec in)
{
	double m2 = 1.0 / 25.0;
	double m1 = 1.0 / 9.0;
	mVec mask5 = { {m2,m2,m2,m2,m2},{m2,m2,m2,m2,m2},{m2,m2,m2,m2,m2},{m2,m2,m2,m2,m2},{m2,m2,m2,m2,m2} };
	mVec mask3 = { {m1,m1,m1} ,{m1,m1,m1}, {m1,m1,m1} };
	vVec out1 = filter(in, mask3);
	out1 = filter(out1, mask3);
	vVec out2 = filter(in, mask5);
	mse_psnr(in, out1);
	mse_psnr(in, out2);
	showimg(out1, "2_b_1");
	showimg(out2, "2_b_2");

}
void pros5(vVec in,vVec src)
{
	double m1 = 1.0 / 9.0;
	double m2 = 1.0 / 25.0;
	mVec mask5 = { {m2,m2,m2,m2,m2},{m2,m2,m2,m2,m2},{m2,m2,m2,m2,m2},{m2,m2,m2,m2,m2},{m2,m2,m2,m2,m2} };
	mVec mask3 = { {m1,m1,m1} ,{m1,m1,m1}, {m1,m1,m1} };
	mVec mask = { {-1,-1,-1,-1,-1},{-1,-1,-1,-1,-1},{-1,-1,49,-1,-1},{-1,-1,-1,-1,-1},{-1,-1,-1,-1,-1} };
	mVec maskr3 = { {-1,-1,-1},{-1,17,-1},{-1,-1,-1}};
	vVec out1 = filter(in, mask,"m",25);
	vVec out2 = filter(out1, mask, "m", 25);
	vVec out3 = filter(out2, mask, "m", 25);
	//vVec res = clear(out2,in);

	/*vVec masksrc;
	for (int i = 0; i < in.size(); i++)
	{
		iVec v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			int pi = src[i][j] - in[i][j];
			v1.push_back(pi);
		}
		masksrc.push_back(v1);
	}*/
	Mat dst = Vec2Mat(in);
	namedWindow("sorce", WINDOW_AUTOSIZE);
	imshow("sorce",dst);
	showimg(out3, "2_c");

}
void pros6(vVec in)
{
	mVec gausfilter08 = Gaussian(5, 0.8);
	mVec gausfilter13 = Gaussian(5, 1.3);
	mVec gausfilter20 = Gaussian(5, 2.0);
	vVec out1 = filter(in, gausfilter08);
	vVec out2 = filter(in, gausfilter13);
	vVec out3 = filter(in, gausfilter20);
	showimg(out1, "3_a_08");
	showimg(out2, "3_a_13");
	showimg(out3, "3_a_20");
}
void pros7(vVec in)
{
	vVec out = DoG(in, 5, 0.5, 1.5);
	showimg(out, "3.b");
}
void pros8(vVec in)
{
	double sum = 0.0;
	mVec LoGfilter = LoG(5,0.5,sum);
	mVec mask5 = { { -2, -4, -4, -4, -2},
	{-4, 0, 8, 0, -4},
	{-4, 8, 24, 8, -4},
	{-4, 0, 8, 0, -4},
	{-2, -4, -4, -4, -2} };
	vVec out = filter(in, LoGfilter);
	//vVec res = normalized(out);
	/*vVec res;
	for (int i = 0; i < in.size(); i++)
	{
		iVec v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			int v = in[i][j] - out[i][j];
			v1.push_back(v);
		}
		res.push_back(v1);
	}*/
	showimg(out, "3.c");
}

int main()
{
	vVec p1 = readfile("lighthouse_480x720.raw", 3);
	vVec p2 = readfile("block&white_256.raw", 1);
	vVec p3 = readfile("chessboard_256.raw", 1);
	vVec p4 = readfile("lena512.raw", 2);
	vVec p5 = readfile("lena512_blurring.raw", 2);
	vVec p6 = readfile("baboon_256.raw", 1);


	char num1, num2, num3,num4;
	while (true)
	{
		printf("1	--		1.Edge Detection.\r\n");
		printf("2	--		2.Mean Filter.\r\n");
		printf("3	--		3.		.\r\n");
		printf("Select function: ");
		scanf(" %c", &num1);
		switch (num1)
		{
		case'1':
			printf("1	--		1.Laplacian filter.\r\n");
			printf("2	--		2.	Sobel filter	.\r\n");
			printf("Select function: ");
			scanf(" %c", &num2);
			switch (num2)
			{
			case'1':
				pros1(p1);
				break;
			case'2':
				pros2(p1);
				break;
			}
			break;
		case'2':
			printf("1	--		1.mean filter	.\r\n");
			printf("2	--		2.	3x3 twice&5x5x	once	.\r\n");
			printf("3	--		3.	sharp	.\r\n");
			printf("Select function: ");
			scanf(" %c", &num3);
			switch (num3)
			{
			case'1':
				pros3(p2,p3);
				break;
			case'2':
				pros4(p4);
				break;
			case'3':
				pros5(p5,p4);
				break;
			}
			break;
		case'3':
			printf("1	--		1.	gaussian filter 5*5.\r\n");
			printf("2	--		2.		.\r\n");
			printf("3	--		3.		.\r\n");
			printf("Select function: ");
			scanf(" %c", &num4);
			switch (num4)
			{
			case'1':
				pros6(p6);
				break;
			case'2':
				pros7(p6);
				break;
			case'3':
				pros8(p6);
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