#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <highgui.hpp>
#include <iostream>
#include <time.h>
#include <imgproc.hpp>
#include <core.hpp>
#include <cstring> 

#define height		256
#define width		256
#define height2		512
#define width2		512

using namespace cv;
using namespace std;
typedef unsigned char  BYTE;

int truncate(int value)				//修正overflow，因char 範圍是-128~127，unsigned char範圍為0~255
														//超過255的話會超過8bit，ex. 256=>100000000(9bit)，溢位會捨去最左邊的bit
														//所以會變成00000000(8bit) =>0。
{
	if (value > 255)		return 255;
	if (value < 0)			return 0;

	return value;
}	   

BYTE out[height][width];
void findpixel(BYTE in[height][width])
{
	int row,col;
	printf("Which row:");
	scanf("%d", &row);
	printf("Which column:");
	scanf("%d", &col);
	if (((row - 1) < 0) || ((col - 1) < 0))
	{
		printf("pixel: Null\n");
	}
	else
	{
		printf("pixel: %d\n", in[row - 1][col - 1]);
	}

}

void pixelvalue(unsigned char path[65536])
{
	int in;
	printf("Find pixel: ");
	scanf("%d", &in);
	printf("pixel vaiues: %d\n", path[in - 1]);

}

void showlena(BYTE in[height][width])
{
	Mat dst(height, width, CV_8UC1,in);
	namedWindow("lena256", WINDOW_AUTOSIZE);
	imshow("lena256", dst);
	imwrite("lena256.png", dst);
}

void rot(BYTE in[height][width],FILE* fp = NULL)
{	
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			out[j][i] = in[i][j];	// 丟入正確pixel
		}
	}

	fp = fopen("lena256rot.raw", "wb");
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			fwrite(&out[i][j], 1, 1, fp);
		}
	}

	Mat dst(height, width, CV_8UC1, &out);
	namedWindow("lena256rot", WINDOW_AUTOSIZE);
	imshow("lena256rot", dst);
	imwrite("lena256rot.png", dst);
}

void randswap(BYTE in[height][width], FILE * fp = NULL)
{
	vector <int> v;
	srand(time(0));
	for (int i = 1; i < 9; i++)
	{
		v.push_back(i);
	}
	random_shuffle(v.begin(), v.end());
	int c = 0;
	for (int k = 0; k < 8; k++)
	{
		int max = v[k] * 32;
		int min = (v[k] - 1) * 32;
		for (int i = min; i < max; i++)
		{
			for (int j = 0; j < width; j++)
			{
				out[c][j] = in[i][j];
			}
			c++;
		}
	}

	fp = fopen("lena256swap.raw", "wb");
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			fwrite(&out[i][j], 1, 1, fp);
		}
	}
	
	Mat dst(height, width, CV_8UC1, &out);
	namedWindow("lena256randswap", WINDOW_AUTOSIZE);
	imshow("lena256randswap", dst);
	imwrite("lena256swap.png", dst);
	
}

void add60(BYTE in[height][width],FILE *fp = NULL)
{

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			out[i][j] = truncate(in[i][j] + 60);

		}
	}
	fp = fopen("lena256_60.raw", "wb");
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			fwrite(&out[i][j], 1, 1, fp);
		}
	}
	Mat dst(height, width, CV_8UC1, &out);
	namedWindow("lena256_60", WINDOW_AUTOSIZE);
	imshow("lena256_60", dst);
	imwrite("lena256add.png", dst);
}

void draw512(BYTE in[height2][width2])
{
	Mat dst512(height2, width2, CV_8UC1, in);
	ellipse(dst512, Point(264, 310), Size(198, 130), 180, 0, 180, Scalar(255, 255, 255), 4, 8);
	putText(dst512, "108318047", Point(80, 300), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 0), 3, false);
	line(dst512, Point(66, 310), Point(460, 310), Scalar(255, 255, 255), 4, 8);
	imwrite("lena512.png", dst512);
	namedWindow("lena512", WINDOW_AUTOSIZE);
	imshow("lena512", dst512);
}


int main()
{
	FILE* fp = NULL;
	FILE* fp2 = NULL;

	BYTE B[height][width];
	BYTE B5[height2][width2];
	BYTE *ptr,*ptr2;

	unsigned char path1[65536];

	int i, j;
	char fr;
	//讀檔
	//printf("Input the raw image path:");
	//scanf("%s",path);
	fp = fopen("lena256.raw", "rb");
	if (fp == NULL)
	{
		printf("can not open the raw image ");
		return 0;
	}
	else
	{
		printf("read 256 OK !\n");
	}
	
	fp2 = fopen("lena_512.raw", "rb");
	if (fp2 == NULL)
	{
		printf("can not open the raw image ");
		return 0;
	}
	else
	{
		printf("read 512 OK !\n");
	}


	// 分配内存并将图像读到二维数组中    	
	ptr = (BYTE*)malloc(width * height * sizeof(BYTE));		//malloc 代表 memory allocation，用來配置指定大小的記憶體空間，
																										//傳回新空間第一個位元組的記憶體位址，配置的空間處於尚未初始化的狀態。
	int c1 = 0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			fread(ptr, 1, 1, fp);
			B[i][j] = *ptr;
			path1[c1] = *ptr;
			ptr++;
			c1++;
		}
	}
	fclose(fp);

	ptr2 = (BYTE*)malloc(width2 * height2 * sizeof(BYTE));		//malloc 代表 memory allocation，用來配置指定大小的記憶體空間，
																										//傳回新空間第一個位元組的記憶體位址，配置的空間處於尚未初始化的狀態。
	for (i = 0; i < height2; i++)
	{
		for (j = 0; j < width2; j++)
		{
			fread(ptr2, 1, 1, fp2);
			B5[i][j] = *ptr2;	
			ptr2++;
		}
	}
	fclose(fp2);

	printf("1	--		1.2.b.(1)	find the intensity of pixel at the [ ]th row []th column.\r\n");
	printf("2	--		1.2.b.(2)	what is the intensity value of [ ]th pixel stored. \r\n");
	printf("3	--		1.2.c	display lena256.raw.\r\n");
	printf("4	--		1.2.d	reloacte and rotate. \r\n");
	printf("5	--		1.2.e	randomly swap. \r\n");
	printf("6	--		1.3.a	add60.\r\n");
	printf("7	--		2	draw graph.\r\n");
	printf("e	--		esc.\r\n");
	
	while (true)
	{
		printf("Which function:");
		scanf(" %c",	&fr);
		//if (fr.getlengh >) { break; }
		switch (fr)
		{
		case '1':
			findpixel(B);
			break;
		case '2':
			pixelvalue(path1);
			break;
		case '3':
			showlena(B);
			break;
		case '4':
			rot(B,fp);
			break;
		case '5':
			randswap(B,fp);
			break;
		case '6':
			add60(B,fp);
			break;
		case '7':
			draw512(B5);
			break;
		case'e':
			exit(0);
		}
		fflush(stdin);
		waitKey(0);
		fclose(fp);
		destroyAllWindows();
	}
	fclose(fp);
	fclose(fp2);
	destroyAllWindows();
}
