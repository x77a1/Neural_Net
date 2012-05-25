/*
 * 	NN.h
 *	Description : It contains a 3 layer Neural network class and back-propagation function
 *  Created on	: Mar 11, 2012
 *  Author	  	: x77a1
 */

#ifndef NN_H_
#define NN_H_
#define sigma(x) ( ((double)(1)) / (((double)(1)) + exp(-x)) )
#include"Dat.h"
class NeuralNet
{
	public:

		int no_of_input_layer , no_of_hidden_layer , no_of_output_layer;
		double inputLayer[1000] , outputLayer[1000] , hiddenLayer[1000];
		double weight0[1000][1000] , weight1[1000][1000];
		double deltaweight0[1000][1000] , deltaweight1[1000][1000];
		double delta[1000];
		double learningRate;
		NeuralNet(int , int , int , double , double);
		void init(int , int , int);
		void dump(int , int , int , double);
};

void NeuralNet::init( int I , int H , int O)
{
	ifstream ifs( "log" , ios::in);
	inputLayer[I] = 1;
	hiddenLayer[H] = 1;
	double x;
	ifs >> x;
	for(int i = 0; i <= I; ++i )
		for(int j = 0; j < H; ++j)
		{	ifs >> weight0[i][j] ; deltaweight0[i][j] = 0; }
	for(int i = 0; i <= H; ++i )
		for(int j = 0; j < O; ++j)
		{ 	ifs >> weight1[i][j] ; deltaweight1[i][j] = 0; }
}

void NeuralNet::dump( int I , int H , int O , double E)
{
	ofstream ifs( "log" , ios::out);
	ifs << E << "\n" ;
	for(int i = 0; i <= I; ++i )
	{
		for(int j = 0; j < H; ++j)
			ifs << weight0[i][j] << " ";
		ifs << "\n";
	}
	for(int i = 0; i <= H; ++i )
	{
		for(int j = 0; j < O; ++j)
			ifs << weight1[i][j] << " ";
		ifs << "\n";
	}
	ifs.close();
}

NeuralNet::NeuralNet(int cil , int chl , int col , double lr , double momentum)
{
	learningRate = lr;
	no_of_hidden_layer = chl;
	no_of_input_layer = cil;
	no_of_output_layer = col;
	Data *Dat = new Data();
	int Epoch = 1000;
	// Initialization

	init( no_of_input_layer , no_of_hidden_layer , no_of_output_layer );

	// TRAINING OF DATA
	while( Epoch--)
	{
	if( (1000 - Epoch) % 100 == 0 )
		learningRate /= 2 , momentum /= 2;
	learningRate = max( learningRate , 0.0000005 );
	momentum = max( momentum , 0.000000001 );
	for(int trainData = 0 ; trainData < Dat->trainSize ; ++trainData)
	{

	for(int i = 0; i < no_of_input_layer; ++i)
	{
		inputLayer[i] =( Dat->trainInp[trainData][i] >= 128 )?1:0;
	}

	for(int j = 0; j < no_of_hidden_layer; ++j)
	{
		hiddenLayer[j] = 0;
		for(int i = 0; i <= no_of_input_layer; ++i)
			hiddenLayer[j] += inputLayer[i] * weight0[i][j];
		hiddenLayer[j] = sigma(hiddenLayer[j]);
	}

	for(int j = 0; j < no_of_output_layer; ++j)
	{
			outputLayer[j] = 0;
			for(int i = 0; i <= no_of_hidden_layer; ++i)
				outputLayer[j] += hiddenLayer[i] * weight1[i][j];
			outputLayer[j] = sigma(outputLayer[j]);
	}

	// APPLYING BACKPROPAGATION

	for(int i = 0; i < no_of_output_layer; ++i)
	{
		if( Dat -> trainOut[trainData] == i)
			delta[i] =  ( 1 - outputLayer[i]) * outputLayer[i] * ( 1 - outputLayer[i]);
		else
			delta[i] =  ( 0 - outputLayer[i]) * outputLayer[i] * ( 1 - outputLayer[i]);
	}

	double tmp;
	for(int i = 0; i <= no_of_input_layer ; ++i)
			for(int h = 0 ; h < no_of_hidden_layer; ++h)
			{
				tmp = 0;
				for (int o = 0; o < no_of_output_layer; ++o)
					tmp += ( delta[o] * weight1[h][o]);
				tmp *= (learningRate * hiddenLayer[h] * ( 1 - hiddenLayer[h]) * inputLayer[i]);
				weight0[i][h] += ( tmp + momentum * deltaweight0[i][h] );
				deltaweight0[i][h] = ( tmp + momentum * deltaweight0[i][h] );
			}
	for(int h = 0; h <= no_of_hidden_layer ; ++h)
			for(int o = 0 ; o < no_of_output_layer; ++o)
			{
				weight1[h][o] += momentum * deltaweight1[h][o] + learningRate * delta[o] * hiddenLayer[h];
				deltaweight1[h][o] = momentum * deltaweight1[h][o] + learningRate * delta[o] * hiddenLayer[h];
			}

	}

	Dat -> randomize();
	}
	// TESTING DATA
	{
	int correctEvalution = 0;
	double totalE = 0;
	for(int testData = 0 ; testData < Dat->testSize ; ++testData)
	{

	for(int i = 0; i < no_of_input_layer; ++i)
		inputLayer[i] = ( Dat->testInp[testData][i] >= 128)?1:0 ;

	for(int j = 0; j < no_of_hidden_layer; ++j)
	{
		hiddenLayer[j] = 0;
		for(int i = 0; i <= no_of_input_layer; ++i)
			hiddenLayer[j] += inputLayer[i] * weight0[i][j];
		hiddenLayer[j] = sigma(hiddenLayer[j]);
	}
	for(int j = 0; j < no_of_output_layer; ++j)
	{
			outputLayer[j] = 0;
			for(int i = 0; i <= no_of_hidden_layer; ++i)
				outputLayer[j] += hiddenLayer[i] * weight1[i][j];
			outputLayer[j] = sigma(outputLayer[j]);
	}
	double E = 0;
	for(int i = 0; i < no_of_output_layer; ++i)
	{
		if( Dat -> testOut[testData] == i)
			E +=  ( 1 - outputLayer[i]) *  ( 1 - outputLayer[i]);
		else
			E +=  ( outputLayer[i]) * outputLayer[i] ;
	}
	E /= 2;
	if( E < 1e-5)
		++correctEvalution;
	totalE += E;

	}

	totalE /= Dat -> testSize ;
	cout << "============================================\n";
	cout << "Mean Square Root Error : " << totalE  << "\n";
	cout << "% correctness          : " << (double)correctEvalution / Dat -> testSize << "\n";
	cout << "============================================\n";
	dump( no_of_input_layer , no_of_hidden_layer , no_of_output_layer , learningRate);
	}

}
#endif /* NN_H_ */
