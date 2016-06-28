/* Neil Babson
   January 24, 2016
   CS 545
   Homework #2

   Header file for neural network program. */

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <iomanip>
#include <cmath>

const float eta = .03;       // learning rate
const float alpha = .03; // momentum
const int N = 300;       // number of hidden layer nodes
const int training_iterations = 1000;  // maximum number of training epocs
using namespace std;

struct instance          // one instance of the data set
{
   int letter;
   float attributes[16];
};

