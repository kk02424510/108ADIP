#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <vector>
#include <opencv.hpp>

using namespace std;
using namespace cv;
typedef unsigned char BYTE;

#define height1 512
#define width1  512
#define height2 256
#define width2  256

vector<vector<BYTE>> readfile(const char* _FileName,int a)
{
	int height, width;
	if (a == 1) { height = height1; width = width1; }
	if (a == 2) { height = height2; width = width2; }
	FILE* fp = NULL;
	vector<vector<BYTE>> out;
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
		vector<BYTE> v1;
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
vector<vector<BYTE>> Negitive(vector<vector<BYTE>> in)
{
	vector<vector<BYTE>> out;
	int height = in.size();
	int width = in[0].size();

	for (int i = 0; i < height; i++)
	{
		vector<BYTE>v1;
		for (int j = 0; j < width; j++)
		{
			int n = 255 - in[i][j];
			v1.push_back(n);
		}
		out.push_back(v1);
	}
	return out;
}
Mat Vec2Mat(vector<vector<BYTE>>array,int a = 0)
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
vector<vector<BYTE>> Mat2Vec(Mat in,int row,int col)
{
	vector<vector<BYTE>> out;
	for (int i = 0; i < row; ++i)
	{
		vector<BYTE>v1;
		for (int j = 0; j < col; ++j)
		{
			v1.push_back(in.at<BYTE>(i,j));
		}
		out.push_back(v1);
	}
	return out;
}
vector<BYTE> BuildTableSDR()
{
	vector<BYTE> table;
	int i;
	float L;
	for (i = 0; i < 256; i++)
	{
		L = (i / 255.0);
		if (L < 0.018){	table.push_back(BYTE((4.5 * L)*255));	}
		else { table.push_back(BYTE((1.099 * pow(L, 0.45) - 0.099) * 255)); }
	}
	return table;
}
vector<BYTE> BuildTableHDR()
{
	vector<BYTE> table;
	int i;
	float L,a,b,c;
	//float aa = 1.0F / 12.0F;
	a = 0.17883277F;
	b = 0.28466892F;
	c = 0.55991073F;
	for (i = 0; i < 256; i++)
	{
		L = (i / 255.0);
		if (L <=1 &&L>(1.0F/12.0F)) { table.push_back(BYTE((a*log(12*L-b)+c)*255)); }
		else { table.push_back(BYTE((sqrt(3)*pow(L,0.5)) * 255)); }
	}
	return table;
}

vector<vector<BYTE>> Bitplane(vector<vector<BYTE>> in,int a)
{
	vector<vector<BYTE>> p;
	for (int i = 0; i < in.size(); i++)
	{
		vector<BYTE> v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			int bi = pow(2, a);
			int put = (in[i][j] & bi)<<(7-a);
			put = put * 255 / 128;
			v1.push_back(put);
		}
		p.push_back(v1);
	}
	return p;
}
vector<vector<BYTE>> Bit_in(vector<vector<BYTE>> in, vector<vector<BYTE>> draw)
{
	vector<vector<BYTE>> p;
	for (int i = 0; i < in.size(); i++)
	{
		vector<BYTE> v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			int put = ((in[i][j]) >> 1)<<1;
			put = put |(draw[i][j]>>7);
			v1.push_back(put);
		}
		p.push_back(v1);
	}
	return p;
}
void showimg(vector<vector<BYTE>> in,const char* _WindowName,int a = 0 )
{
	Mat dst = Vec2Mat(in,a);
	namedWindow(_WindowName, WINDOW_AUTOSIZE);
	imshow(_WindowName, dst);
	char str1[30] = {0};
	strncpy_s(str1, _WindowName,20);
	char str2[] = ".png";
	strcat(str1,str2);
	char* c;
	c = const_cast<char*>(str1);
	imwrite(c, dst);
}
void pros1(vector<vector<BYTE>>in)
{
	vector<vector<BYTE>>out = Negitive(in);
	showimg(out, "lena_512negitive");
}
void pros2(vector<vector<BYTE>>in)
{
	vector<vector<BYTE>>SDR_out,HDR_out;
	vector<BYTE> SDR_table = BuildTableSDR();
	vector<BYTE> HDR_table = BuildTableHDR();

	for (int i = 0; i < in.size(); i++)
	{
		vector<BYTE>v1,v2;
		for (int j = 0; j < in[0].size(); j++)
		{
			int pixelSDR = SDR_table[(in[i][j])];
			v1.push_back(pixelSDR);
			int pexilHDR = HDR_table[(in[i][j])];
			v2.push_back(pexilHDR);
		}
		SDR_out.push_back(v1);
		HDR_out.push_back(v2);
	}
	showimg(SDR_out, "SDR_output");
	showimg(HDR_out, "HDR_output");

}
void findqu(vector<vector<BYTE>>in)
{
	vector<vector<BYTE>> p0 = Bitplane(in, 0);
	vector<vector<BYTE>> p1 = Bitplane(in, 1);
	
	showimg(p0, "babbon_bit0");
	showimg(p1, "babbon_bit1");

	vector<vector<BYTE>> pf;
	for (int i = 0; i < in.size(); i++)
	{
		vector<BYTE>v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			 v1.push_back(p0[i][j] & p1[i][j]);
		}
		pf.push_back(v1);
	}
	showimg(pf, "Q");
}
void answer(vector<vector<BYTE>>in)
{
	vector<vector<BYTE>> p0 = Bitplane(in,0);
	Mat dst = Vec2Mat(p0);
	putText(dst, "108318047", Point(80, 200), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 0), 3, false);
	putText(dst, "Kuan-Yu,Su", Point(70, 400), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 0), 3, false);
	namedWindow("lena_0bit", WINDOW_AUTOSIZE);
	imshow("lena_0bit", dst);
	imwrite("ans.png", dst);
	vector<vector<BYTE>>draw = Mat2Vec(dst,512,512);
	vector<vector<BYTE>> res = Bit_in(in, draw);
	showimg(res, "res");
}
vector<int> Histogram(vector<vector<BYTE>>in)
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
	return hist;
}
vector<vector<BYTE>> Hist_eq(vector<vector<BYTE>>in)
{
	vector<vector<BYTE>>out;
	vector<int> hist = Histogram(in);
	vector<double> cdf;
	vector<int>hist_eq;
	double c = 0;
	double size = in.size() * in[0].size();
	//double min = *min_element(cdf.begin(), cdf.end());
	for (int i = 0; i < 256; i++)
	{
		c += (double)hist[i];
		hist_eq.push_back(c / size * 255);
	}
	for (int i = 0; i < in.size(); i++)
	{
		vector<BYTE> v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			v1.push_back(hist_eq[in[i][j]]);
		}
		out.push_back(v1);
	}
	return out;
}
Mat drawhist(vector<int>hist)
{
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
	return histmap;
}
vector<vector<BYTE>> Hist_Match(vector<vector<BYTE>>in1, vector<vector<BYTE>>in2)
{
	vector<vector<BYTE>> out;
	vector<int> hist_lena = Histogram(in1);
	vector<int> hist_dog = Histogram(in2);
	Mat dst = drawhist(hist_dog);
	namedWindow("hist_in", WINDOW_AUTOSIZE);
	imshow("hist_in", dst);
	vector<double> cdf_dog,cdf_lena;
	vector<int> map(256) ;
	double clena = 0,cdog = 0;
	double size1 = in1.size() * in1[0].size();
	double size2 = in2.size() * in2[0].size();
	for (int i = 0; i < 256; i++)
	{
		clena += (double)hist_lena[i] / size1;
		cdog += (double)hist_dog[i]/size2;
		cdf_dog.push_back(cdog);
		cdf_lena.push_back(clena);
	}
	double diffA = 0, diffB = 0;
	int k = 0;
	for (int i = 0; i < 256; i++)
	{
		diffB = 1;
		for (int j = k; j < 256; j++)
		{
			diffA = abs(cdf_lena[i] - cdf_dog[j]);
			if (diffA - diffB < 1.0E-08)
			{
				diffB = diffA;
				k = j;
			}
			else {
				k = abs(j - 1);
				break;
			}
		}
		if (k == 255) {
			for (int l = i; l < 256; l++) {
				map[l] = k;
			}
			break;
		}
		map[i] = k;
	}
	Mat dst2 = drawhist(hist_lena);
	namedWindow("hist_re", WINDOW_AUTOSIZE);
	imshow("hist_re", dst2);
	imwrite("hist_lena.png", dst2);
	for (int i = 0; i < in1.size(); i++)
	{
		vector<BYTE> v1;
		for (int j = 0; j < in1[0].size(); j++)
		{
			v1.push_back(map[in1[i][j]]);
		}
		out.push_back(v1);
	}
	return out;
}

void pros3(vector<vector<BYTE>>in,int a)
{
	vector<int>hist = Histogram(in);
	Mat histmap = drawhist(hist);
	if (a == 1)
	{
		namedWindow("hist_dogbright", WINDOW_AUTOSIZE);
		imshow("hist_dogbright", histmap);
		imwrite("hist_dogbright.png", histmap);
	}
	if (a == 2)
	{
		namedWindow("hist_dogdark", WINDOW_AUTOSIZE);
		imshow("hist_dogdark", histmap);
		imwrite("hist_dogdark.png", histmap);
	}
	waitKey(0);
}
void pros4(vector<vector<BYTE>>in,int a)
{
	vector<vector<BYTE>>eq = Hist_eq(in);
	vector<int> hist = Histogram(eq);
	Mat histmap = drawhist(hist);
	if (a == 1)
	{
		showimg(eq, "dogbright_eq");
		namedWindow("hist_dogbright", WINDOW_AUTOSIZE);
		imshow("hist_dogbright", histmap);
		imwrite("hist_bright_eq.png",histmap);
	}
	if (a == 2)
	{
		showimg(eq, "dogdark_eq");
		namedWindow("hist_dogdark", WINDOW_AUTOSIZE);
		imshow("hist_dogdark", histmap);
		imwrite("hist_dark_eq.png", histmap);
	}
}
void pros5(vector<vector<BYTE>>in1, vector<vector<BYTE>>in2)
{
	vector<vector<BYTE>> res = Hist_Match(in1, in2);
	vector<int> hist = Histogram(res);
	Mat dst = drawhist(hist);
	namedWindow("hist", WINDOW_AUTOSIZE);
	imshow("hist", dst);
	imwrite("matched.png", dst);
	showimg(res, "lena_match");

}

int main()
{
	vector<vector<BYTE>>p1 = readfile("lena_512.raw",1);
	vector<vector<BYTE>>p2 = readfile("baboonQ_256.raw",2);
	vector<vector<BYTE>>p3 = readfile("dog_bright256.raw", 2);
	vector<vector<BYTE>>p4 = readfile("dog_dark256.raw", 2);
	
	
	char num1,num2, num3;
	while (true)
	{
		printf("1	--		1.		Grey Level Transformation.\r\n");
		printf("2	--		2.		Bit Plane.\r\n");
		printf("3	--		3.		Histogram.\r\n");
		printf("Select function: ");
		scanf(" %c", &num1);
		switch (num1)
		{
		case'1':
			printf("1. Negitive \r\n");
			printf("2. SDR/HDR Curve \r\n");
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
			findqu(p2);
			answer(p1);
			break;
		case'3':
			printf("1. Histogram dark and bright \r\n");
			printf("2. Histogram Equalization \r\n");
			printf("3. Histogram Match \r\n");
			printf("Select function: ");
			scanf(" %c", &num3);
			switch (num3)
			{
			case'1':
				pros3(p3, 1);
				pros3(p4, 2);
				break;
			case'2':
				pros4(p3 ,1);
				pros4(p4, 2);
				break;
			case'3':
				pros5(p1,p3);
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