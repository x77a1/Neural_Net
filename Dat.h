//	This is header file for functions related to FILE HANDLING
#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<cstdlib>
#include<cstdio>
#include<cmath>
using namespace std;
int c2i( unsigned char c )
{
	int a = 0 , z = 1;
	while( c )
	{
		if( c & 1 )
			a |= z;
		z <<=1;
		c >>=1;
	}
	return a;
}
int atox( char * buff )
{
	int z = 1;
	int res= 0;
	for(int i = 3; i >= 0; --i)
	{
		int a = c2i( buff[i] );
		res += ( a * z );
		z *= 256;
	}
	return res;
}
class Data
{
	public:
	Data();
	int x[60000][28 * 28] , y[60000];
	int trainInp[60000][28 * 28] , trainOut[60000];
	int testInp[10000][28 * 28] , testOut[10000];
	int testSize , trainSize ;
	void generate();
	void randomize();
	void getData( char * , char *);
};
Data::Data()
{
	testSize = 0;
	trainSize = 0;
	generate();
}
void Data::getData( char  fileName[] , char label[] )
{
	ifstream file_image ( fileName , ios::in|ios::binary );
	char buff[5];
	char buf;
	int magic_number = 0 ;
	int count;
	if( file_image.is_open() )
	{
		file_image.get( buff , 5 );
		magic_number = atox( buff );
		int row , col;
		file_image.get( buff , 5 );
		count  = atox( buff );
		file_image.get( buff , 5 );
		row = atox (buff);
		file_image.get( buff , 5 );
		col = atox (buff);
		for(int i = 0; i < count ;++i)
		{
			for(int j = 0; j < 28 * 28; ++j)
			{
				file_image.get( buf ) ;
				x[i][j] = c2i(buf);
			}
		}
	}
	file_image.close();
	ifstream file ( label , ios::in|ios::binary );
	if( file.is_open() )
	{
		file.get( buff , 5 );
		magic_number = atox( buff );
		file.get( buff , 5 );
		count  = atox( buff );
		for(int i = 0; i < count ;++i)
		{
				file.get( buf ) ;
				y[i] = c2i(buf);
		}
	}
	file.close();
}
void Data::generate()
{
	char image[] = "train-images.idx3-ubyte";
	char label[] = "train-labels.idx1-ubyte";
	getData( image, label);
	int Y = 28 * 28 ;
	for(int i = 0; i < 60000; ++i)
	{
		for(int j = 0; j < Y; ++j)
			trainInp[trainSize][j] = x[i][j];
		trainOut[trainSize++] = y[i];
	}
	char tesimage[] = "t10k-images.idx3-ubyte";
	char teslabel[] = "t10k-labels.idx1-ubyte";
	getData( tesimage , teslabel);
	for(int i = 0; i < 10000; ++i)
	{
		for(int j = 0; j < Y; ++j)
			testInp[testSize][j] = x[i][j];
		testOut[testSize++] = y[i];
	}
	cout << "=================================\n";
	cout << "Reading Data from file finished\n";
	cout << "=================================\n";
}
void Data::randomize()
{
	for(int i = 0; i < 60000; ++i)
	{
		int j = rand() % 60000;
		int t;
		for(int k = 0; k < 28 * 28; ++k)
		{
			t = trainInp[i][k];
			trainInp[i][k] = trainInp[j][k];
			trainInp[j][k] = t;
		}
		t = trainOut[i];
		trainOut[i] = trainOut[j];
		trainOut[j] = t;
	}
	cout << "\n=================================\n";
	cout << "Data set Randomized for new epoch\n";
	cout << "=================================\n";
}
