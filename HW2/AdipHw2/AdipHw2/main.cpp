#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv.hpp>
#include <vector>
#include <queue>

using namespace std;
using namespace cv;
typedef unsigned char BYTE;
typedef unsigned short SHORT;

#define height1	 256
#define width1	 256
#define hz 20
#define wz 20

typedef struct myNode{
	int x;
	int y;
}myNode;
myNode** path;


vector<vector<BYTE>> readfile(const char* _FileName, int a)
{
	int height, width;
	if (a == 1) { height = height1; width = width1; }
	if (a == 2) { height = hz; width = wz; }
	if (a == 3) { height = 512; width = 512; }
	//if (a == 1) { height = height2; width = width2; }
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
Mat Vec2Mat(vector<vector<BYTE>>array)
{
	int row = array.size();
	int col = array[0].size();
	Mat img(row, col, CV_8UC1);
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

//--v--		write raw image	--v-- 
void wraw(vector<vector<BYTE>>in,const char *_FileName)
{
	

	FILE* fp = NULL;
	fp = fopen(_FileName,"wb");
	for (int i = 0; i < in.size(); i++)
	{
		for (int j = 0; j < in[0].size(); j++)
		{
			fwrite(&in[i][j],1,1, fp);
		}
	}
	
	fclose(fp);
}		

//--v--		zooming		--v--
void zoom(vector<vector<BYTE>> in,int a)
{
	int height, width;
	if (a == 1) { height = height1; width = width1; }
	if (a == 2) { height = 512; width = 512; }
	vector<vector<BYTE> > pz;
	for (int i = 0; i < height; i++)
	{
		for (int a = 0; a < 2; a++)
		{
			vector<BYTE> v1;
			for (int j = 0; j < width; j++)
			{
				for (int b = 0; b < 2; b++)
				{
					v1.push_back(in[i][j]);
				}
			}
			pz.push_back(v1);
		}
	}
	Mat dst = Vec2Mat(pz);
	if (a == 1)
	{
		namedWindow("zooming256", WINDOW_AUTOSIZE);
		imshow("zooming256", dst);
		//wraw(pz,"lena512.raw");
		imwrite("lena256zoom2.png", dst);
	}
	if (a == 2)
	{
		namedWindow("zooming512", WINDOW_AUTOSIZE);
		imshow("zooming512", dst);
		//wraw(pz,"lena512.raw");
		imwrite("lena512*2.png", dst);
	}
	
}

//--v--		shrinking		--v--
void shrink(vector<vector<BYTE>> in,const char *Type)
{
	vector<vector<BYTE>> ps;
	for (int i = 0; i < height1; i=i+2)
	{
		vector<BYTE> v1;
		for (int j = 0; j < width1; j=j+2)
		{
			v1.push_back(in[i][j]);
		}
		ps.push_back(v1);
	}

	Mat dst = Vec2Mat(ps);
	
	if (Type == "noblur")
	{
		namedWindow("shrinking", WINDOW_NORMAL);
		imshow("shrinking", dst);
		//wraw(ps,"lena128.raw");
		imwrite("lena128.png", dst);
	}
	if (Type == "blur")
	{
		namedWindow("shrinking-blur", WINDOW_NORMAL);
		imshow("shrinking-blur", dst);
		//wraw(ps, "lena128b.raw");
		imwrite("lena128b.png", dst);
	}
}

vector<vector<BYTE>> nearest_neighbor(vector<vector<BYTE>> in,float aa)
{
	float m = 1.0 / aa;
	vector<vector<BYTE>> out;
	int px, py;
	for (int i = 0; i < in.size()*aa; i++)
	{
		vector<BYTE>	v1;
		for (int j = 0; j < in[0].size() * aa; j++)
		{
			px = int(i * m);
			py = int(j * m);

			v1.push_back(in[px][py]);
		}
		out.push_back(v1);
	}
	return out;
}

vector<vector<BYTE>> Bilinear(vector<vector<BYTE>> in,float aa)
{
	vector<vector<BYTE>> out;
	float	m = (float)(in.size() * aa) / (float)(in.size() - 1);
	float	a, b;
	int x, y,pi;

	for (int i = 0; i < in.size() * aa; i++)
	{
		vector<BYTE> v1;
		for (int j = 0; j < in[0].size() * aa; j++)
		{
			y	=	i / m;
			x	=	j / m;
			a	=	((float)i - (float)y * m) / m;
			b	= ((float)j - (float)x * m) / m;
			pi = (1 - a) * (1 - b) * in[y][x] + a * (1 - b) * in[y][x + 1] + b * (1 - a) * in[y + 1][x] + a * b * in[y + 1][x + 1];
			v1.push_back(pi);
		}
		out.push_back(v1);
	}
	return out;
}

void getW_x(float w_x[4], float x) {
	int X = (int)x;
	float a = -0.5;
	float stemp_x[4];
	
	stemp_x[0] = 1 + (x - X);
	stemp_x[1] = x - X;
	stemp_x[2] = 1 - (x - X);
	stemp_x[3] = 2 - (x - X);

	w_x[0] = a * abs(stemp_x[0] * stemp_x[0] * stemp_x[0]) - 5 * a * stemp_x[0] * stemp_x[0] + 8 * a * abs(stemp_x[0]) - 4 * a;
	w_x[1] = (a + 2) * abs(stemp_x[1] * stemp_x[1] * stemp_x[1]) - (a + 3) * stemp_x[1] * stemp_x[1] + 1;
	w_x[2] = (a + 2) * abs(stemp_x[2] * stemp_x[2] * stemp_x[2]) - (a + 3) * stemp_x[2] * stemp_x[2] + 1;
	w_x[3] = a * abs(stemp_x[3] * stemp_x[3] * stemp_x[3]) - 5 * a * stemp_x[3] * stemp_x[3] + 8 * a * abs(stemp_x[3]) - 4 * a;
}

void getW_y(float w_y[4], float y) {
	float a = -0.5;
	int Y = (int)y;
	float stemp_y[4];
	stemp_y[0] = 1.0 + (y - Y);
	stemp_y[1] = y - Y;
	stemp_y[2] = 1 - (y - Y);
	stemp_y[3] = 2 - (y - Y);

	w_y[0] = a * abs(stemp_y[0] * stemp_y[0] * stemp_y[0]) - 5 * a * stemp_y[0] * stemp_y[0] + 8 * a * abs(stemp_y[0]) - 4 * a;
	w_y[1] = (a + 2) * abs(stemp_y[1] * stemp_y[1] * stemp_y[1]) - (a + 3) * stemp_y[1] * stemp_y[1] + 1;
	w_y[2] = (a + 2) * abs(stemp_y[2] * stemp_y[2] * stemp_y[2]) - (a + 3) * stemp_y[2] * stemp_y[2] + 1;
	w_y[3] = a * abs(stemp_y[3] * stemp_y[3] * stemp_y[3]) - 5 * a * stemp_y[3] * stemp_y[3] + 8 * a * abs(stemp_y[3]) - 4 * a;
}

vector<vector<BYTE>> Bicubic (vector<vector<BYTE>> in,float a)
{
	vector<vector<BYTE>> out;
	float Row_out = in.size() * a;
	float Col_out = in[0].size() * a;
	for (int i = 0; i < Row_out; i++)
	{
		vector<BYTE> v1;
		for (int j = 0; j < Col_out; j++)
		{
			float x = i * (in.size() / Row_out);
			float y = j * (in[0].size() / Col_out);
			float w_x[4], w_y[4];
			getW_x(w_x, x);
			getW_y(w_y, y);
			int temp = 0;
			for ( int s = 0 ; s <= 3; s++) {
				for ( int t = 0 ; t <= 3; t++) {
					int u = int(x) + s - 1;
					int v = int(y) + t - 1;
					if (u < 0) { u = 0; };
					if (v < 0) { v = 0; };
					if (u >= in.size()) { u = in.size() - 1; }
					if (v >= in[0].size()) { v = in[0].size() - 1; }
					temp = temp + in[u][v]* w_x[s] * w_y[t];
				}
			}
			v1.push_back(temp);
		}
		out.push_back(v1);
	}
	
	//wraw(out, "lena_bicubic.raw");
	return out;
}

vector<vector<BYTE>> Quantizing(vector<vector<BYTE>> in, int a)
{
	vector<vector<BYTE>> out;
	for (int i = 0; i < in.size() ; i++)
	{
		vector<BYTE> v1;
		for (int j = 0; j < in[0].size() ; j++)
		{
			int shift = (in[i][j] >> a);
			shift = shift * 255 /( (pow(2, (8-a)) - 1));
			v1.push_back(shift);
		}
		out.push_back(v1);
	}
	return out;
}

void mse_psnr(vector<vector<BYTE>>in1, vector<vector<BYTE>>in2,double &mse,double &psnr)
{
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
}

vector<vector<BYTE>> Edge(vector<vector<BYTE>>in, int a,int d)
{
	int da, db;
	if (d == 'r') { da = 0; db = 1; }
	if (d == 'l') { da = 1; db = 0; }
	vector<vector<BYTE>> out;
	for (int i = 0; i < in[0].size()-da; i++)
	{
		vector<BYTE> v1;
		for (int j = 0; j < in.size() - db; j++)
		{
			int left = in[i][j];
			int right = in[i+da][j+db];
			int dif = right - left;
			if (abs(dif) > a)
			{
				v1.push_back(255);
			}
			else
			{
				v1.push_back(0);
			}
		}
		out.push_back(v1);
	}
	return out;
}

vector<vector<BYTE>> maze85(vector<vector<BYTE>>in)
{
	vector<vector<BYTE>>maze85;
	for (int i = 0; i < in[0].size(); i++)
	{
		vector<BYTE> v1;
		for (int j = 0; j < in.size(); j++)
		{
			if (in[i][j] ==85 ) { v1.push_back(255); }
			else { v1.push_back(0); }
		}
		maze85.push_back(v1);
	}
	return maze85;
}
vector<vector<BYTE>> maze170(vector<vector<BYTE>>in)
{
	vector<vector<BYTE>>maze;
	for (int i = 0; i < in[0].size(); i++)
	{
		vector<BYTE> v1;
		for (int j = 0; j < in.size(); j++)
		{
			if ((in[i][j] == 85) || (in[i][j] == 170)) { v1.push_back(255); }
			else { v1.push_back(0); }
		}
		maze.push_back(v1);
	}
	return maze;
}
vector<vector<BYTE>> maze255(vector<vector<BYTE>>in)
{
	vector<vector<BYTE>>maze;
	for (int i = 0; i < in[0].size(); i++)
	{
		vector<BYTE> v1;
		for (int j = 0; j < in.size(); j++)
		{
			if (in[i][j] == 85 || in[i][j] == 170 || in[i][j] == 255) { v1.push_back(255); }
			else { v1.push_back(0); }
		}
		maze.push_back(v1);
	}
	return maze;
}

void BFS(myNode start, myNode end,int a, vector<vector<BYTE>> maze, myNode* movein)
{
	path = new myNode * [20];
	for (int i = 0; i <= 20; i++)
	{
		path[i] = new myNode[20];
	}
	int i;
	myNode tempN;
	myNode tempNM;
queue<myNode>Q;
	Q.push(start);
	vector<vector<BYTE>> flag = maze;
	flag[start.x][start.y] = 0;
	while (Q.size()!=0)
	{
		tempN = Q.front();
		Q.pop();
		for (i = 0; i < a; i++)
		{
			tempNM.x = tempN.x + movein[i].x;
			tempNM.y = tempN.y + movein[i].y;
			if (tempNM.x >= 0 && tempNM.y >= 0 && tempNM.x <= end.x && tempNM.y <= end.y && maze[tempNM.x][tempNM.y] == 255 && flag[tempNM.x][tempNM.y] != 0)
			{

				Q.push(tempNM);
				flag[tempNM.x][tempNM.y] = 0;
				path[tempNM.x][tempNM.y] = tempN;
				//printf("(%d,%d)", tempNM.x, tempNM.y);
			}
		}
	}
	return ;
}

void BFSDm(myNode start, myNode end, vector<vector<BYTE>> maze)
{
	myNode movein[4] = { { 1,0 }, { -1,0 }, { 0,1 }, { 0,-1 } };
	myNode move8[8] = { { 1,-1},{1,1},{-1,1},{-1,-1} };
	path = new myNode * [50];
	for (int i = 0; i <= 50; i++)
	{
		path[i] = new myNode[50];
	}

	int i,count;
	bool cf = false;
	myNode tempN;
	myNode tempNM;
	myNode chack;
	queue<myNode>Q;
	Q.push(start);
	vector<vector<int>> flag;
	vector<vector<int>> flag1;
	for (int i = 0; i < maze.size(); i++)
	{
		vector<int>v1;
		for (int j = 0; j < maze[0].size(); j++)
		{
			v1.push_back(0);
		}
		flag.push_back(v1);
		flag1.push_back(v1);
	}
		

	flag[start.x][start.y] = 1;
	flag1[start.x][start.y] = 0;

	while (Q.size() != 0)
	{
		/*if (Q.back().x == 19 && Q.back().y == 19)
		{
			break;
		}*/
		tempN = Q.front();
		printf(" P0:(%d,%d)\t", tempN.x, tempN.y);
		count = 0;
		Q.pop();
		printf("have been find: %d", flag[tempN.x][tempN.y]);
		for (i = 0; i < 4; i++)
		{
			tempNM.x = tempN.x + movein[i].x;
			tempNM.y = tempN.y + movein[i].y;
			
			/*if ((i == 0 || i == 2) && flag[tempNM.x][tempNM.y] == 1)
			{
				flag[tempNM.x][tempNM.y] = 0;
			}*/
			if ((i==0||i==2)&&tempNM.x >= 0 && tempNM.y >= 0 && tempNM.x <= end.x && tempNM.y <= end.y && flag1[tempNM.x][tempNM.y] == 1)
			{
				flag[tempNM.x][tempNM.y] = 0;
				flag1[tempNM.x][tempNM.y] == 2;
			}
			if (tempNM.x >= 0 && tempNM.y >= 0 && tempNM.x <= end.x && tempNM.y <= end.y && maze[tempNM.x][tempNM.y] == 255 && flag[tempNM.x][tempNM.y] == 0)
			{
				
				printf(" P4find:(%d,%d)", tempNM.x, tempNM.y);
				cf = true;
				Q.push(tempNM);
				flag[tempNM.x][tempNM.y] =1;
				chack = path[tempNM.x][tempNM.y];
				
				path[tempNM.x][tempNM.y] = tempN;
				/*if (tempNM.x==19&& tempNM.y ==19)
				{
					while (Q.size() != 0)
					{
						Q.pop();
					}
					break;
				}*/
			}
			if (cf == false)
			{
				count++;
			}
			else 
			{
				count=0;
				cf = false;
			}
		}
		
		if (count == 4)
		{
			for (i = 0; i < 4; i++)
			{
				tempNM.x = tempN.x + move8[i].x;
				tempNM.y = tempN.y + move8[i].y;
				//if (flag[tempN.x][tempN.y] != 1)
				//{
					if (tempNM.x >= 0 && tempNM.y >= 0 && tempNM.x <= end.x && tempNM.y <= end.y && maze[tempNM.x][tempNM.y] == 255 && flag[tempNM.x][tempNM.y] ==0)
					{
						printf(" \t P8find:(%d,%d)", tempNM.x, tempNM.y);
						count = 0;
						cf = false;
						Q.push(tempNM);
						flag[tempNM.x][tempNM.y] = 1;
						flag1[tempNM.x][tempNM.y] = 1 ;
						path[tempNM.x][tempNM.y] = tempN;
						

					}
				//}
				
			}
		}
		printf("\r\n");
	}

	return;
}


vector<vector<BYTE>> outputpath(myNode end,vector<vector<BYTE>> in,int &c )
{
	vector<vector<BYTE>> pro;
	myNode temp = end;
	myNode t = { 100,100 };
	
	if (end.x == 0 && end.y == 0)
	{
		printf("(%d,%d)", end.x, end.y);
		c++;
		return in;
	}
	else
	{

		printf("(%d,%d)", temp.x, temp.y);
		c++ ;
		temp = path[temp.x][temp.y];
		pro=outputpath(temp,in,c);
		pro[temp.x][temp.y] = 0;
		//printf("(%d,%d)", temp.x, temp.y);
		
	}
	
	return pro;

}

void Distance(vector<vector<BYTE>>in,int a,int b)
{
	myNode move4[4] = { { 1,0 }, { -1,0 }, { 0,1 }, { 0,-1 } };
	myNode move8[8] = { { 1,0 }, { -1,0 }, { 0,1 }, { 0,-1 },{ 1,-1},{1,1},{-1,1},{-1,-1} };
	myNode start = {0,0};
	myNode end = { 19,19 };
	vector<vector<BYTE>> vout;
	for (int i = 0; i < in[0].size(); i++)
	{
		vector<BYTE> v1;
		for (int j = 0; j < in.size(); j++)
		{
			v1.push_back(255);
		}
		vout.push_back(v1);
	}
	if (a == 4)
	{
		BFS(start, end, 4, in, move4);
		int count = 0;
		vout = outputpath(end, vout, count);
		printf("step = %d", count);
		vout[19][19] = 0;
		printf("(19, 19)");
		printf("finish!!\r\n");
		Mat dst2 = Vec2Mat(vout);
		if (b == 85)
		{
			namedWindow("D4_85", WINDOW_NORMAL);
			imshow("D4_85", dst2);
			imwrite("D4_85.png", dst2);
		}
		if (b == 170)
		{
			namedWindow("D4_170", WINDOW_NORMAL);
			imshow("D4_170", dst2);
			imwrite("D4_170.png", dst2);
		}
		if (b == 255)
		{
			namedWindow("D4_255", WINDOW_NORMAL);
			imshow("D4_255", dst2);
			imwrite("D4_255.png", dst2);
		}
	}
	if (a == 8)
	{
		BFS(start, end, 8, in, move8);
		int count = 0;
		vout = outputpath(end, vout, count);
		printf("step = %d", count);
		vout[19][19] = 0;
		printf("(19, 19)");
		printf("finish!!\r\n");
		Mat dst2 = Vec2Mat(vout);
		if (b == 85)
		{
			namedWindow("D8_85", WINDOW_NORMAL);
			imshow("D8_85", dst2);
			imwrite("D8_85.png", dst2);
		}
		if (b == 170)
		{
			namedWindow("D8_170", WINDOW_NORMAL);
			imshow("D8_170", dst2);
			imwrite("D8_170.png", dst2);
		}
		if (b == 255)
		{
			namedWindow("D8_255", WINDOW_NORMAL);
			imshow("D8_255", dst2);
			imwrite("D8_255.png", dst2);
		}
	}
	if (a == 0)
	{
		BFSDm(start, end, in);
		int count = 0;
		vout = outputpath(end, vout, count);
		printf("step = %d", count);
		vout[19][19] = 0;
		printf("(19, 19)");
		printf("finish!!\r\n");
		Mat dst2 = Vec2Mat(vout);
		if (b == 85)
		{
			namedWindow("Dm_85", WINDOW_NORMAL);
			imshow("Dm_85", dst2);
			imwrite("Dm_85.png", dst2);
		}
		if (b == 170)
		{
			namedWindow("Dm_170", WINDOW_NORMAL);
			imshow("Dm_170", dst2);
			imwrite("Dm_170.png", dst2);
		}
		if (b == 255)
		{
			namedWindow("Dm_255", WINDOW_NORMAL);
			imshow("Dm_255", dst2);
			imwrite("Dm_255.png", dst2);
		}
	}

	
	
	
	
}

void step1	(vector<vector<BYTE>>in)
{
	vector<vector<BYTE>> lena_NearestNeighbor1 = nearest_neighbor(nearest_neighbor(in, 3.5), 0.5);
	vector<vector<BYTE>> lena_NearestNeighbor2 = nearest_neighbor(nearest_neighbor(in, 0.5), 3.5);
	vector<vector<BYTE>> lena_NearestNeighbor3 = nearest_neighbor(in, 1.75);
	namedWindow("lena_NearestNeighbor1", WINDOW_AUTOSIZE);
	namedWindow("lena_NearestNeighbor2", WINDOW_AUTOSIZE);
	namedWindow("lena_NearestNeighbor3", WINDOW_AUTOSIZE);
	Mat dst1 = Vec2Mat(lena_NearestNeighbor1);
	Mat dst2 = Vec2Mat(lena_NearestNeighbor2);
	Mat dst3 = Vec2Mat(lena_NearestNeighbor3);
	imshow("lena_NearestNeighbor1", dst1);
	imshow("lena_NearestNeighbor2", dst2);
	imshow("lena_NearestNeighbor3", dst3);
	//wraw(lena_NearestNeighbor1, "lena_NearestNeighbor1.raw");
	//wraw(lena_NearestNeighbor2, "lena_NearestNeighbor2.raw");
	//wraw(lena_NearestNeighbor3, "lena_NearestNeighbor3.raw");
	imwrite("lena_NearestNeighbor1.png", dst1);
	imwrite("lena_NearestNeighbor2.png", dst2);
	imwrite("lena_NearestNeighbor3.png", dst3);
}
void step2	(vector<vector<BYTE>>in)
{
	vector<vector<BYTE>> lena_Bilinear1 = Bilinear(Bilinear(in, 3.5), 0.5);
	vector<vector<BYTE>> lena_Bilinear2 = Bilinear(Bilinear(in, 0.5), 3.5);
	vector<vector<BYTE>> lena_Bilinear3 = Bilinear(in, 1.75);
	namedWindow("lena_Bilinear1", WINDOW_AUTOSIZE);
	namedWindow("lena_Bilinear2", WINDOW_AUTOSIZE);
	namedWindow("lena_Bilinear3", WINDOW_AUTOSIZE);
	Mat dst1 = Vec2Mat(lena_Bilinear1);
	Mat dst2 = Vec2Mat(lena_Bilinear2);
	Mat dst3 = Vec2Mat(lena_Bilinear3);
	imshow("lena_Bilinear1", dst1);
	imshow("lena_Bilinear2", dst2);
	imshow("lena_Bilinear3", dst3);
	//wraw(lena_Bilinear1, "lena_Bicubic1.raw");
	//wraw(lena_Bilinear2, "lena_Bicubic2.raw");
	//wraw(lena_Bilinear3, "lena_Bicubic3.raw");
	imwrite("lena_Bilinear1.png", dst1);
	imwrite("lena_Bilinear2.png", dst2);
	imwrite("lena_Bilinear3.png", dst3);
}
void step3	(vector<vector<BYTE>>in)
{
	vector<vector<BYTE>> lena_Bicubic1 = Bicubic(Bicubic(in, 3.5), 0.5);
	vector<vector<BYTE>> lena_Bicubic2 = Bicubic(Bicubic(in, 0.5), 3.5);
	vector<vector<BYTE>> lena_Bicubic3 = Bicubic(in, 1.75);
	namedWindow("lena_Bicubic1", WINDOW_AUTOSIZE);
	namedWindow("lena_Bicubic2", WINDOW_AUTOSIZE);
	namedWindow("lena_Bicubic3", WINDOW_AUTOSIZE);
	Mat dst1	=	Vec2Mat(lena_Bicubic1);
	Mat dst2	= Vec2Mat(lena_Bicubic2);
	Mat dst3	= Vec2Mat(lena_Bicubic3);
	imshow("lena_Bicubic1", dst1);
	imshow("lena_Bicubic2", dst2);
	imshow("lena_Bicubic3", dst3);
	//wraw(lena_Bicubic1, "lena_Bicubic1.raw");
	//wraw(lena_Bicubic2, "lena_Bicubic2.raw");
	//wraw(lena_Bicubic3, "lena_Bicubic3.raw");
	imwrite("lena_Bicubic1.png", dst1);
	imwrite("lena_Bicubic2.png", dst2);
	imwrite("lena_Bicubic3.png", dst3);
}
void step41	(vector<vector<BYTE>>in)
{
	vector<vector<BYTE>> lena_7bit = Quantizing(in, 1);
	vector<vector<BYTE>> lena_6bit = Quantizing(in, 2);
	vector<vector<BYTE>> lena_5bit = Quantizing(in, 3);
	vector<vector<BYTE>> lena_4bit = Quantizing(in, 4);
	vector<vector<BYTE>> lena_3bit = Quantizing(in, 5);
	vector<vector<BYTE>> lena_2bit = Quantizing(in, 6);
	vector<vector<BYTE>> lena_1bit = Quantizing(in, 7);
	Mat dst1 = Vec2Mat(lena_7bit);
	Mat dst2 = Vec2Mat(lena_6bit);
	Mat dst3 = Vec2Mat(lena_5bit);
	Mat dst4 = Vec2Mat(lena_4bit);
	Mat dst5 = Vec2Mat(lena_3bit);
	Mat dst6 = Vec2Mat(lena_2bit);
	Mat dst7 = Vec2Mat(lena_1bit);
	namedWindow("lena_7bit", WINDOW_AUTOSIZE);
	namedWindow("lena_6bit", WINDOW_AUTOSIZE);
	namedWindow("lena_5bit", WINDOW_AUTOSIZE);
	namedWindow("lena_4bit", WINDOW_AUTOSIZE);
	namedWindow("lena_3bit", WINDOW_AUTOSIZE);
	namedWindow("lena_2bit", WINDOW_AUTOSIZE);
	namedWindow("lena_1bit", WINDOW_AUTOSIZE);
	imshow("lena_7bit", dst1);
	imshow("lena_6bit", dst2);
	imshow("lena_5bit", dst3);
	imshow("lena_4bit", dst4);
	imshow("lena_3bit", dst5);
	imshow("lena_2bit", dst6);
	imshow("lena_1bit", dst7);
	imwrite("lena_7bit.png", dst1);
	imwrite("lena_6bit.png", dst2);
	imwrite("lena_5bit.png", dst3);
	imwrite("lena_4bit.png", dst4);
	imwrite("lena_3bit.png", dst5);
	imwrite("lena_2bit.png", dst6);
	imwrite("lena_1bit.png", dst7);
	/*wraw(lena_7bit, "lena_7bit.raw");
	wraw(lena_6bit, "lena_6bit.raw");
	wraw(lena_5bit, "lena_5bit.raw");
	wraw(lena_4bit, "lena_4bit.raw");
	wraw(lena_3bit, "lena_3bit.raw");
	wraw(lena_2bit, "lena_2bit.raw");
	wraw(lena_1bit, "lena_1bit.raw");*/
	double mse, psnr;
	double data[7][2];
	mse_psnr(in, lena_7bit, mse, psnr);		data[0][0] = mse;		data[0][1] = psnr;
	mse_psnr(in, lena_6bit, mse, psnr);		data[1][0] = mse;		data[1][1] = psnr;
	mse_psnr(in, lena_5bit, mse, psnr);		data[2][0] = mse;		data[2][1] = psnr;
	mse_psnr(in, lena_4bit, mse, psnr);		data[3][0] = mse;		data[3][1] = psnr;
	mse_psnr(in, lena_3bit, mse, psnr);		data[4][0] = mse;		data[4][1] = psnr;
	mse_psnr(in, lena_2bit, mse, psnr);		data[5][0] = mse;		data[5][1] = psnr;
	mse_psnr(in, lena_1bit, mse, psnr);		data[6][0] = mse;		data[6][1] = psnr;
	
	
	FILE* fp;

	fp = fopen("lena_mse.txt", "wb");
	for (int i = 0; i < 7; i++)
	{
		fprintf(fp,"%d.\t",i+1);
		for (int j = 0; j < 2; j++)
		{
			if (j == 0)
			{
				fprintf(fp, "mse: %.2f \t", data[i][j]);
			}
			if (j == 1)
			{
				fprintf(fp, "psnr: %.2f \r\n", data[i][j]);
			}
		}
	}
	fclose(fp);
}
void step42	(vector<vector<BYTE>>in)
{
	vector<vector<BYTE>> baboon_7bit = Quantizing(in, 1);
	vector<vector<BYTE>> baboon_6bit = Quantizing(in, 2);
	vector<vector<BYTE>> baboon_5bit = Quantizing(in, 3);
	vector<vector<BYTE>> baboon_4bit = Quantizing(in, 4);
	vector<vector<BYTE>> baboon_3bit = Quantizing(in, 5);
	vector<vector<BYTE>> baboon_2bit = Quantizing(in, 6);
	vector<vector<BYTE>> baboon_1bit = Quantizing(in, 7);
	Mat dst1 = Vec2Mat(baboon_7bit);
	Mat dst2 = Vec2Mat(baboon_6bit);
	Mat dst3 = Vec2Mat(baboon_5bit);
	Mat dst4 = Vec2Mat(baboon_4bit);
	Mat dst5 = Vec2Mat(baboon_3bit);
	Mat dst6 = Vec2Mat(baboon_2bit);
	Mat dst7 = Vec2Mat(baboon_1bit);
	namedWindow("baboon_7bit", WINDOW_AUTOSIZE);
	namedWindow("baboon_6bit", WINDOW_AUTOSIZE);
	namedWindow("baboon_5bit", WINDOW_AUTOSIZE);
	namedWindow("baboon_4bit", WINDOW_AUTOSIZE);
	namedWindow("baboon_3bit", WINDOW_AUTOSIZE);
	namedWindow("baboon_2bit", WINDOW_AUTOSIZE);
	namedWindow("baboon_1bit", WINDOW_AUTOSIZE);
	imshow("baboon_7bit", dst1);
	imshow("baboon_6bit", dst2);
	imshow("baboon_5bit", dst3);
	imshow("baboon_4bit", dst4);
	imshow("baboon_3bit", dst5);
	imshow("baboon_2bit", dst6);
	imshow("baboon_1bit", dst7);
	imwrite("baboon_7bit.png", dst1);
	imwrite("baboon_6bit.png", dst2);
	imwrite("baboon_5bit.png", dst3);
	imwrite("baboon_4bit.png", dst4);
	imwrite("baboon_3bit.png", dst5);
	imwrite("baboon_2bit.png", dst6);
	imwrite("baboon_1bit.png", dst7);
	/*wraw(baboon_7bit, "baboon_7bit.raw");
	wraw(baboon_6bit, "baboon_6bit.raw");
	wraw(baboon_5bit, "baboon_5bit.raw");
	wraw(baboon_4bit, "baboon_4bit.raw");
	wraw(baboon_3bit, "baboon_3bit.raw");
	wraw(baboon_2bit, "baboon_2bit.raw");
	wraw(baboon_1bit, "baboon_1bit.raw");*/

	double mse, psnr;
	double data[7][2];
	mse_psnr(in, baboon_7bit, mse, psnr);		data[0][0] = mse;		data[0][1] = psnr;
	mse_psnr(in, baboon_6bit, mse, psnr);		data[1][0] = mse;		data[1][1] = psnr;
	mse_psnr(in, baboon_5bit, mse, psnr);		data[2][0] = mse;		data[2][1] = psnr;
	mse_psnr(in, baboon_4bit, mse, psnr);		data[3][0] = mse;		data[3][1] = psnr;
	mse_psnr(in, baboon_3bit, mse, psnr);		data[4][0] = mse;		data[4][1] = psnr;
	mse_psnr(in, baboon_2bit, mse, psnr);		data[5][0] = mse;		data[5][1] = psnr;
	mse_psnr(in, baboon_1bit, mse, psnr);		data[6][0] = mse;		data[6][1] = psnr;


	FILE* fp;

	fp = fopen("baboon_mse.txt", "wb");
	for (int i = 0; i < 7; i++)
	{
		fprintf(fp, "%d.\t", i + 1);
		for (int j = 0; j < 2; j++)
		{
			if (j == 0){fprintf(fp, "mse: %.2f \t", data[i][j]);}
			if (j == 1){fprintf(fp, "psnr: %.2f \r\n", data[i][j]);}
		}
	}
	fclose(fp);
}
void step5		(vector<vector<BYTE>>in1, vector<vector<BYTE>>in2,int a,const char d)
{
	if (d == 'r')
	{
		if (a == 25)
		{
			vector<vector<BYTE>> lena_edge = Edge(in1, 25,'r');
			vector<vector<BYTE>> baboon_edge = Edge(in2, 25,'r');
			Mat dst1 = Vec2Mat(lena_edge);
			Mat dst2 = Vec2Mat(baboon_edge);
			namedWindow("lena_edge_right25", WINDOW_AUTOSIZE);
			namedWindow("baboon_edge_right25", WINDOW_AUTOSIZE);
			imshow("lena_edge_right25", dst1);
			imshow("baboon_edge_right25", dst2);
			imwrite("lena_edge_right25.png", dst1);
			imwrite("baboon_edge_right25.png", dst2);
			//wraw(lena_edge, "lena_edge_right25.raw");
			//wraw(baboon_edge, "baboon_edge_right25.raw");
		}
		if (a == 50)
		{
			vector<vector<BYTE>> lena_edge = Edge(in1, 50,'r');
			vector<vector<BYTE>> baboon_edge = Edge(in2, 50,'r');
			Mat dst1 = Vec2Mat(lena_edge);
			Mat dst2 = Vec2Mat(baboon_edge);
			namedWindow("lena_edge_right50", WINDOW_AUTOSIZE);
			namedWindow("baboon_edge_right50", WINDOW_AUTOSIZE);
			imshow("lena_edge_right50", dst1);
			imshow("baboon_edge_right50", dst2);
			imwrite("lena_edge_right50.png", dst1);
			imwrite("baboon_edge_right50.png", dst2);
			//wraw(lena_edge, "lena_edge_right50.raw");
			//wraw(baboon_edge, "baboon_edge_right50.raw");
		}
	}
	if (d == 'l')
	{
		vector<vector<BYTE>> lena_edge = Edge(in1, 25, 'l');
		vector<vector<BYTE>> baboon_edge = Edge(in2, 25, 'l');
		Mat dst1 = Vec2Mat(lena_edge);
		Mat dst2 = Vec2Mat(baboon_edge);
		namedWindow("lena_edge_lower25", WINDOW_AUTOSIZE);
		namedWindow("baboon_edge_lower25", WINDOW_AUTOSIZE);
		imshow("lena_edge_lower25", dst1);
		imshow("baboon_edge_lower25", dst2);
		imwrite("lena_edge_lower25.png", dst1);
		imwrite("baboon_edge_lower25.png", dst2);
		//wraw(lena_edge, "lena_edge_lower25.raw");
		//wraw(baboon_edge, "baboon_edge_lower25.raw");
	}
	
}
void step6		(vector<vector<BYTE>>in,int a,int b)
{
	if (a == 85)
	{
		Mat dst1 = Vec2Mat(maze85(in));
		namedWindow("maze_20_85", WINDOW_NORMAL);
		imshow("maze_20_85", dst1);
		Distance(maze85(in),b,a);
		imwrite("maze_20_85.png", dst1);
	}
	if (a == 170)
	{
		Mat dst1 = Vec2Mat(maze170(in));
		namedWindow("maze_20_170", WINDOW_NORMAL);
		imshow("maze_20_170", dst1);
		Distance(maze170(in),b,a);
		imwrite("maze_20_170.png", dst1);
	}
	if (a == 255)
	{
		Mat dst1 = Vec2Mat(maze255(in));
		namedWindow("maze_20_255", WINDOW_NORMAL);
		imshow("maze_20_255", dst1);
		Distance(maze255(in),b,a);
		imwrite("maze_20_255.png", dst1);
	}
}
int main()
{
	vector<vector<BYTE>>p1 = readfile("lena_256.raw",1);
	vector<vector<BYTE>>p2 = readfile("lena_256_b.raw",1);
	vector<vector<BYTE>>p3 = readfile("baboon_256.raw",1);
	vector<vector<BYTE>>p4 = readfile("maze_20.raw",2);
	//vector<vector<BYTE>>p5 = readfile("lena_512.raw",3);


	printf("1	--		1.(a)		Zooming.\r\n");
	printf("2	--		1.(b)		Shrinking.\r\n");
	printf("3	--		1.(c)		compare.\r\n");
	printf("4	--		2.		8bit->1bit.\r\n");
	printf("5	--		3.		edge.\r\n");
	printf("6	--		4.		distance and parh.\r\n");


	char num1,num2,num3,num4,num5,num6,num7,num8;
	while (true)
	{
		printf("select function: ");
		scanf("	%c", &num1);
		switch (num1)
		{
		case '1':
			zoom(p1,1);
			break;
		case '2':
			shrink(p1,"noblur");
			waitKey(0);
			shrink(p2,"blur");
			break;
		case'3':
			printf("1.	nearest neighbor .\r\n");
			printf("2.	bilinear .\r\n");
			printf("3.	bicubic .\r\n");
			printf("select: ");
			scanf("	%c", &num2);
			switch (num2)
			{
			case'1':
				step1(p1);
				break;
			case'2':
				step2(p1);
				break;
			case'3':
				step3(p1);
				break;
			}
			break;
		case '4':
			printf("1. lena\r\n");
			printf("2. baboon\r\n");
			printf("select: ");
			scanf("	%c", &num3);
			switch (num3)
			{
			case '1':
				step41(p1);
				break;
			case'2':
				step42(p3);
				break;
			}
			break;
		case'5':
			printf("1. var = 25-right \r\n");
			printf("2. var = 50-right \r\n");
			printf("1. var = 25-lower \r\n");
			printf("select: ");
			scanf("	%c", &num4);
			switch (num4)
			{
			case'1':
				step5(p1, p3,25,'r');
				break;
			case'2':
				step5(p1, p3, 50,'r');
				break;
			case'3':
				step5(p1, p3, 25, 'l');
				break;
			}
			break;
		case'6':
			printf("1. D4 \r\n");
			printf("2. D8 \r\n");
			printf("3. Dm \r\n");
			printf("select: ");
			scanf("	%c", &num5);
			switch (num5)
			{
			case'1':
				printf("1. var = 85 \r\n");
				printf("2. var = 170 \r\n");
				printf("3. var = 255 \r\n");
				printf("select: ");
				scanf("	%c", &num6);
				switch (num6)
				{
				case'1':
					step6(p4, 85, 4);
					break;
				case'2':
					step6(p4, 170, 4);
					break;
				case'3':
					step6(p4, 255, 4);
					break;
				}
				break;
			case'2':
				printf("1. var = 85 \r\n");
				printf("2. var = 170 \r\n");
				printf("3. var = 255 \r\n");
				printf("select: ");
				scanf("	%c", &num7);
				switch (num7)
				{
				case'1':
					step6(p4, 85, 8);
					break;
				case'2':
					step6(p4, 170, 8);
					break;
				case'3':
					step6(p4, 255, 8);
					break;
				}
				break;
			case'3':
				printf("1. var = 85 \r\n");
				printf("2. var = 170 \r\n");
				printf("3. var = 255 \r\n");
				printf("select: ");
				scanf("	%c", &num8);
				switch (num8)
				{
				case'1':
					step6(p4, 85, 0);
					break;
				case'2':
					step6(p4, 170, 0);
					break;
				case'3':
					step6(p4, 255, 0);
					break;
				}
				break;
			}
			break;
		case '0':
			exit(0);
		}
		fflush(stdin);
		waitKey(0);
		destroyAllWindows();
	}
}