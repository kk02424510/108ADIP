#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <opencv.hpp>
#include <vector>
#include <cmath>
#include <complex>

using namespace std;
using namespace cv;

typedef unsigned char BYTE;
typedef vector<BYTE> iVec;
typedef vector<iVec>vVec;
typedef vector<double> dVec;
typedef vector<dVec>ddVec;

vVec readfile(const char* _FileName, int height, int width)
{
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
int format(int in)
{
	if (in > 255) { in = 255; }
	if (in < 0) { in = 0; }
	return in;
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
vVec dim_mask(int size)
{
	vVec out(size, iVec(size,0));
	for (int i = 0; i < size; i++)
	{
		for (int j = abs(size/2-i); j < abs(size / 2 - i)+(size-abs(size/2-i)*2);j++)
		{
			out[i][j] = 1;
		}
	}
	return out;
}
vVec dilate(vVec in,int size)
{
	vVec out(in);
	int m = size / 2;
	for (int i = 0; i < in.size(); i++)
	{
		iVec v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			if (in[i][j] > 0)
			{
				for (int u = -m; u <= m; u++)
				{
					for (int v = -m; v <= m; v++)
					{
						out[i+u][j+v] = 255;
					}
				}
			}

		}
	}
	return out;
}
vVec dilate_dim(vVec in, int size)
{
	vVec out(in);
	int m = size / 2;
	for (int i = 0; i < in.size(); i++)
	{
		iVec v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			if (in[i][j] > 0)
			{
				for (int u = -m; u <= m; u++)
				{
					for (int v = -(size / 2 - abs(u)); v < -(size / 2 - abs(u)) + (size - abs(u) * 2); v++)
					{
						out[i + u][j + v] = 255;
					}
				}
			}

		}
	}
	return out;
}
vVec dilate_image(vVec in, int size,int value)
{
	vVec out;
	int m = size / 2;
	for (int i = 0+m; i < in.size()-m; i++)
	{
		iVec v1;
		for (int j = 0+m; j < in[0].size()-m; j++)
		{
			int max = 0;
			for (int u = -m; u <= m; u++)
			{
				for (int v = -m; v <= m; v++)
				{
					if (in[i + u][j + v] > max) { max = in[i + u][j + v]; }
				}
			}
			v1.push_back(format(max + value));
		}
		out.push_back(v1);
	}
	return out;
}
vVec erode(vVec in, int size)
{
	vVec out(in);
	int m = size / 2;
	for (int i = 0+m; i < in.size()-m; i++)
	{
		iVec v1;
		for (int j = 0+m; j < in[0].size()-m; j++)
		{
			if (in[i][j] > 0)
			{
				bool flag = true;
				for (int u = -m; u <= m; u++)
				{
					for (int v = -m; v <= m; v++)
					{
						if (in[i + u][j + v] == 0)
						{
							flag = false;
						}
					}
				}
				if (flag)
				{
					out[i][j] = 255;
				}
				else
				{
					out[i][j] = 0;
				}
			}

		}
	}
	return out;
}
vVec erode_dim(vVec in, int size)
{
	vVec out(in);
	int m = size / 2;
	for (int i = 0 + m; i < in.size() - m; i++)
	{
		iVec v1;
		for (int j = 0 + m; j < in[0].size() - m; j++)
		{
			if (in[i][j] > 0)
			{
				bool flag = true;
				for (int u = -m; u <= m; u++)
				{
					for (int v = -(size / 2 - abs(u)); v < -(size / 2 - abs(u)) + (size - abs(u) * 2); v++)
					{
						if (in[i + u][j + v] == 0)
						{
							flag = false;
						}
					}
				}
				if (flag)
				{
					out[i][j] = 255;
				}
				else
				{
					out[i][j] = 0;
				}
			}

		}
	}
	return out;
}
vVec erode_image(vVec in, int size,int value)
{
	vVec out;
	int m = size / 2;
	for (int i = 0+m; i < in.size()-m; i++)
	{
		iVec v1;
		for (int j = 0+m; j < in[0].size()-m; j++)
		{
			int min = 255;
			for (int u = -m; u <= m; u++)
			{
				for (int v = -m; v <= m; v++)
				{
					if (in[i + u][j + v] < min) { min = in[i + u][j + v]; }
				}
			}
			v1.push_back(format(min - value));
		}
		out.push_back(v1);
	}
	return out;
}
vVec opening(vVec in,int size,int value)
{
	vVec out = dilate_image(erode_image(in, size,value), size,value);
	imshow("opening", Vec2Mat(out));
	return out;
}
vVec closing(vVec in, int size, int value)
{
	vVec out = erode_image(dilate_image(in, size, value), size, value);
	imshow("colsing", Vec2Mat(out));
	return out;
}

void countnum(vVec in)
{
	vVec mask(in.size(), iVec(in[0].size(), 0));
	int count = 10;
	for (int i = 0; i < in.size(); i++)
	{
		for (int j = 0; j < in[0].size(); j++)
		{
			if (mask[i][j] == 0&&in[i][j]==255)
			{
				bool flag = true;
				int n = 0;
				for (int u = -1; u <= 1; u++)
				{
					for (int v = -1; v <= 1; v++)
					{
						if (in[i+u][j+v]==255&&mask[i + u][j + v] != 0)
						{
							flag = false;
							n = mask[i + u][j + v];
						}
					}
				}
				if (!flag)
				{
					mask[i][j] = n;
				}
				if (flag)
				{ 
					mask[i][j] = count;
					count += 10; 
				}
				
			}


		}
	}
	Mat test = Vec2Mat(mask);
}
void non_blod(vVec in)
{
	vVec blod_er3 = erode(in, 3);
	vVec bold_er3_di5 = dilate(blod_er3, 5);
	vVec no_blod;
	for (int i = 0; i < in.size(); i++)
	{
		iVec v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			v1.push_back(in[i][j] - bold_er3_di5[i][j]);
		}
		no_blod.push_back(v1);
	}
	//imshow("no_blod", Vec2Mat(no_blod));
	vVec blod_er33 = erode_dim(in, 5);
	//imshow("blod_er", Vec2Mat(blod_er33));
	vVec non_blod;
	for (int i = 0; i < in.size(); i++)
	{
		iVec v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			v1.push_back(blod_er33[i][j] | no_blod[i][j]);
		}
		non_blod.push_back(v1);
	}
	imshow("res_nonblod", Vec2Mat(non_blod));
	imwrite("1.c.non_blod.png", Vec2Mat(non_blod));
}
void blod(vVec in)
{
	vVec blod_er3 = erode_dim(in, 3);
	vVec bold_er3_di5 = dilate(blod_er3, 5);
	vVec no_blod;
	for (int i = 0; i < in.size(); i++)
	{
		iVec v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			v1.push_back(in[i][j] * (1-bold_er3_di5[i][j]/255));
		}
		no_blod.push_back(v1);
	}
	
	vVec blod_all = dilate_dim(no_blod, 5);

	blod_er3 = dilate_dim(blod_er3, 3);
	for (int i = 0; i < in.size(); i++)
	{
		for (int j = 0; j < in[0].size(); j++)
		{
			blod_all[i][j] = blod_all[i][j]| blod_er3[i][j];
		}
	}
	imshow("All bold", Vec2Mat(blod_all));
	imwrite("1_3_allbold.png", Vec2Mat(blod_all));
}
void pros1a(vVec in)
{
	imshow("src", Vec2Mat(in));
	vVec blod_er = erode(in, 3);
	//imshow("src_er", Vec2Mat(blod_er));
	vVec blod_re = dilate(blod_er,5);
	//imshow("src_er_di", Vec2Mat(blod_re));
	vVec res;
	for (int i = 0; i < in.size(); i++)
	{
		iVec v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			v1.push_back(in[i][j] * (1-blod_re[i][j]/255));
		}
		res.push_back(v1);
	}
	imshow("1.a", Vec2Mat(res));
	imwrite("1_a.png", Vec2Mat(res));
	Mat labels,stats,centroids;
	int nccomps = connectedComponentsWithStats(Vec2Mat(res),labels,stats,centroids,8);
	printf("num: %d\r\n", nccomps-3);
}
void pros1b(vVec in)
{
	imshow("src", Vec2Mat(in));
	vVec blod_er = erode(in, 3);
	vVec blod_re = dilate(blod_er, 3);
	imshow("src_er", Vec2Mat(blod_re));
	imwrite("1_b.png", Vec2Mat(blod_re));
	Mat labels, stats, centroids;
	int nccomps = connectedComponentsWithStats(Vec2Mat(blod_re), labels, stats, centroids, 8);
	printf("num: %d\r\n", nccomps -1);
}
void pros1c(vVec in)
{
	imshow("src", Vec2Mat(in));
	non_blod(in);
	blod(in);
}
void pros2a(vVec in)
{
	Mat src = Vec2Mat(in);
	imwrite("2_a_src.png", Vec2Mat(in));
	Mat src_opening = Vec2Mat(opening(in, 5, 1));
	imwrite("opening.png", src_opening);
	Mat src_closing = Vec2Mat(closing(in, 5, 1));
	imwrite("closing.png", src_closing);
}
void pros2b(vVec in)
{
	Mat src = Vec2Mat(in);
	vVec o1 = opening(mirroredpadding(in, 6),7, 1);
	Mat src_o = Vec2Mat(o1);
	vVec c1 = closing(mirroredpadding(o1, 8), 9, 1);
	Mat src_c = Vec2Mat(c1);
	vVec o2 = opening(mirroredpadding(c1, 30), 31, 1);
	Mat src_o2 = Vec2Mat(o2);
	vVec res;
	for (int i = 0; i < in.size(); i++)
	{
		iVec v1;
		for (int j = 0; j < in[0].size(); j++)
		{
			if (o2[i][j] > 90) { v1.push_back(255); }
			else { v1.push_back(0); }
		}
		res.push_back(v1);
	}
	Mat res_fi = Vec2Mat(res);
	imwrite("2.2.png", res_fi);
}
int main()
{
	vVec p1 = readfile("letters_360x180.raw", 180, 360);
	vVec p2 = readfile("ground_480x609.raw", 609, 480);

	char num1, num2, num3, num4;
	while (true)
	{
		printf("1	--		1.	Binary morphology.\r\n");
		printf("2	--		2.	Grey-level morphology.\r\n");
		printf("Select function: ");
		scanf(" %c", &num1);
		switch (num1)
		{
		case'1':
			printf("a	--		remove bold.\r\n");
			printf("b	--		remove non-bold.\r\n");
			printf("c	--		convert.\r\n");
			printf("Select function: ");
			scanf(" %c", &num2);
			switch (num2)
			{
			case'1':
				pros1a(p1);
				break;
			case'2':
				pros1b(p1);
				break;
			case'3':
				pros1c(p1);
				break;
			}
			break;
		case'2':
			printf("a	--		Opening & Closing.\r\n");
			printf("b	--		Edge.\r\n");
			printf("Select function: ");
			scanf(" %c", &num3);
			switch (num3)
			{
			case'1':
				pros2a(p2);
				break;
			case'2':
				pros2b(p2);
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