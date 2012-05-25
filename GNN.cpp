//============================================================================
// Name        : GNN.cpp
// Author      : x77a1
// Version     : 1.0
// Description : Multiple layer Neural Network
//============================================================================
// FILES INCLUDED AND USAGE
// GNN 		:- TESTING AND TRAINING WITH DATA
// Dat 		:- GENERATE DATA AND RANDOMIZE TRAINING SET
// NN		:- CONTAIN NN LIBRARY ( BACKPROPAGATION )
#include "NN.h"

int main(void) {
	cout << "Neural Network x77a1 \n";
	NeuralNet *NN =  new NeuralNet(28 * 28 , 65 , 10 , 0.00005 , 0.00000001);
	return 0;
}
