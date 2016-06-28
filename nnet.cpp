/* Neil Babson
   January 24, 2016
   CS 545
   Homework #2 */

// This program trains a neural network to recognize letters from the Letter Recognition dataset at the UCI
// machine-learning repository http://archive.ics.uci.edu/ml/datasets/Letter+Recognition.
// The 20,000 item dataset is divided into equal sized training and testing sets which are standardized.
// The network trains until it achievse perfect accuracy catagorizing the training set or it reaches a 
// set maximum number of training epochs.  During each epoch the network updates its weights using back-propogation
// with stochastic gradient descent.  After each training epoch the network checks its prediction accuracy against 
// both the training set and the testing set.

#include "nnet.h"

// Function Prototypes
void load_data(instance training_data[], instance test_data[]);
void standardize_data(instance training_data[], instance test_data[]);
void initialize_weights(float input[][N], float hidden[][26]); 
void shuffle_data(instance data[]);
void train(instance test_data[], instance training_data[], float input[][N], float hidden[][26]); 
float check_accuracy(instance data[], float input[][N], float hidden [][26]);

int main()
{
   instance training_data[10000], test_data[10000];
   float input[17][N];                  // Weights from input layer to hidden layer
   float hidden[N+1][26];               // Weights from hidden layer to output layer

   srand(time(NULL));

   load_data(training_data, test_data);
   standardize_data(training_data, test_data);
   initialize_weights(input, hidden);
   shuffle_data(training_data);
   train(test_data, training_data, input, hidden);

   return 0;
}

// Calculates and returns the network's accuracy on a dataset. 
float check_accuracy(instance data[], float input[][N], float hidden [][26])
{
   float hidden_total[N];
   float output_total[26];
   int prediction, total_correct = 0;
   float highest_activation;

   for (int i = 0; i < 10000; ++i)
   {
      // Initialize input to neurons
      for (int j = 0; j < N; ++j)
	 hidden_total[j] = 0;
      for (int j = 0; j < 26; ++j)
	 output_total[j] = 0;

      // For each hidden node
      for (int j = 0; j < N; ++j)
      {
	 // From each input node
	 for (int k = 0; k < 16; ++k)
	    // Calculate dot product
	    hidden_total[j] += data[i].attributes[k] * input[k][j];
	 // Add in bias
	 hidden_total[j] += input[16][j];
	 // Replace hidden total with sigmoid activation result
	 hidden_total[j] = 1 / (1 +  exp(-hidden_total[j]));
	 // To each output node
	 for (int k = 0; k < 26; ++k)
	 {
	    // Propogate to output layer
	    output_total[k] += hidden[j][k] * hidden_total[j];
	 }
      }
      // Add hidden layer bias
      for (int j = 0; j < 26; ++j)
      {
	 output_total[j] += hidden[N][j];
	 // Replace output_total with sigmoid activation result
	 output_total[j] = 1 / (1 + exp(-output_total[j]));
      }
      highest_activation = 0;
      // Determine network prediction
      for (int j = 0; j < 26; ++j)
	 if (output_total[j] > highest_activation)
	 {
	    prediction = j;
	    highest_activation = output_total[j];
	 }
      if (data[i].letter == prediction)
	 ++total_correct;
   }
   return (float) total_correct / 10000.0;
}

// Trains the network on the training set, updating the weights after every training instance using
// back propogation with stochastic gradient descent.  Training continues until the network achieves 
// perfect accuracy on the test set or until the  number of epochs is equal to the constant 
// 'training iterations'.  After each epoch the network's accuracy on both the training and testing sets
// is calculated and displayed, and the training data is shuffled.
void train(instance test_data[], instance training_data[], float input[][N], float hidden [][26])
{
   float hidden_total[N];     // Activation values of hidden neurons
   float output_total[26];    // Activation values of output neurons
   float highest_activation, target, error_total, change, accuracy;
   int total_correct;
   int prediction, epoch = 1;;
   float hidden_error[N+1];    // Error values for hidden neurons
   float output_error[26];     // Error vaslues for output neurons
   float input_last_change[17][N] = {0};   // Input to hidden layer weights last adjustment
   float hidden_last_change[N+1][26] = {0}; // Hidden to output layer weights last adjustment

   do
   {
      total_correct = 0;
      for (int i = 0; i < 10000; ++i)
      {
	 // Initialize input to neurons
	 for (int j = 0; j < N; ++j)
	    hidden_total[j] = 0;
	 for (int j = 0; j < 26; ++j)
	    output_total[j] = 0;

	 // For each hidden node
	 for (int j = 0; j < N; ++j)
	 {
	    // From each input node
	    for (int k = 0; k < 16; ++k)
	    {
	       // Calculate dot product
	       hidden_total[j] += training_data[i].attributes[k] * input[k][j];
	    }
	    // Add in bias
	    hidden_total[j] += input[16][j];
	    // Replace hidden total with sigmoid activation result
	    hidden_total[j] = 1 / (1 +  exp(-hidden_total[j]));
	    // To each output node
	    for (int k = 0; k < 26; ++k)
	    {
	       // Propogate to output layer
	       output_total[k] += hidden[j][k] * hidden_total[j];
	    }
	 }
	 // Add hidden layer bias
	 for (int j = 0; j < 26; ++j)
	 {
	    output_total[j] += hidden[N][j];
	    // Replace output_total with sigmoid activation result
	    output_total[j] = 1 / (1 + exp(-output_total[j]));
	 }
	 highest_activation = 0;
	 // Determine network prediction
	 for (int j = 0; j < 26; ++j)
	    if (output_total[j] > highest_activation)
	    {
	       prediction = j;
	       highest_activation = output_total[j];
	    }
	 if (training_data[i].letter == prediction)
	    ++total_correct;

	 //Calculate error terms for output nodes
	 for (int j = 0; j < 26; ++j)
	 {
	    if (training_data[i].letter == j)
	       target = .9;
	    else 
	       target = .1;
	    output_error[j] = output_total[j] * (1 - output_total[j]) * (target - output_total[j]);
	 }
	 // Caculate error terms for hidden nodes
	 for (int j = 0; j < N + 1; ++ j)
	 {
	    error_total = 0;
	    for (int k = 0; k < 26; ++k)
	       error_total += hidden[j][k] * output_error[k];
	    hidden_error[j] = hidden_total[j] * (1 - hidden_total[j]) * error_total;
	 }

	 // Update weights from hidden to output layer
	 for (int j = 0; j < N + 1; ++j)
	    for (int k = 0; k < 26; ++k)
	    {
	       change = eta * output_error[k] * hidden_total[j] + alpha * hidden_last_change[j][k];
	       hidden[j][k] += change;
	       hidden_last_change[j][k] = change;
	    }
	 
         // Update hidden bias weights
	 for (int j = 0; j < 26; ++j)
	 {
	    change = eta  * output_error[j] + alpha * hidden_last_change[N][j];
	    hidden[N][j] += change;
	    hidden_last_change[N][j] = change;
	 }

	 // Update weights from input to hidden layer
	 for (int j = 0; j < 16; ++j)
	    for (int k = 0; k < N; ++k)
	    {
	       change = eta * hidden_error[k] * training_data[i].attributes[j] + alpha * input_last_change[j][k];
	       input[j][k] += change;
	       input_last_change[j][k] = change;
	    }
	 // Update input bias weights 
	 for (int j = 0; j < N; ++j)
	 {
	    change = eta * hidden_error[j] + alpha * input_last_change[16][j];
	    input[16][j] += change;
	    input_last_change[16][j] = change;
	 }

      }

      cout << "Epoch " << epoch; 
      accuracy = check_accuracy(training_data, input, hidden);
      cout << "\nAccuracy on training set: " << accuracy;
      cout << "\t\tAccuracy on test set: " << check_accuracy(test_data, input, hidden) << endl;
      ++epoch;
      shuffle_data(training_data);
   }
   while (epoch < training_iterations && accuracy < 1.0);   
}

// Performs the Fisher-Yates shuffle on a data set
void shuffle_data(instance data[])
{
   int j;
   instance temp;

   for (int i = 1; i < 9998; ++i)
   {
      j = rand() % i;
      temp.letter = data[i].letter;
      for (int k = 0; k < 16; ++k)
	 temp.attributes[k] = data[i].attributes[k];
      data[i].letter  = data[i+j].letter;
      for (int k = 0; k < 16; ++k)
	 data[i].attributes[k] = data[i+j].attributes[k];
      data[i+j].letter = temp.letter;
      for (int k = 0; k < 16; ++k)
	 data[i+j].attributes[k] = temp.attributes[k];
   }
}

// Set the network's weights to random values between -.25 and +.25
void initialize_weights(float input[][N], float hidden[][26])
{
   for (int i = 0; i < 17; ++i)
      for (int j = 0; j < N; ++j)
         input[i][j] = rand() / (((float) RAND_MAX) * 2) - .25;
   for (int i = 0; i < (N+1); ++i)
      for (int j = 0; j < 26; ++j)
	 hidden[i][j] = rand () / (((float) RAND_MAX) * 2) - .25;
}

// Standardize the training and test data values
void standardize_data(instance training_data[], instance test_data[])
{
   float mean[16] = {0};
   float dev[16] = {0};

   for (int i = 0; i < 10000; ++i)
      for (int j = 0; j < 16; ++j)
	 mean[j] += training_data[i].attributes[j];
   for (int i = 0; i < 16; ++i)
      mean[i] /= 10000;
   for (int i = 0; i < 10000; ++i)
      for (int j = 0; j < 16; ++j)
         dev[j] += pow((training_data[i].attributes[j] - mean[j]),2);
   for (int i = 0; i < 16; ++i) 
       dev[i] = sqrt(dev[i]/10000);
   for (int i = 0; i < 10000; ++i)
      for (int j = 0; j < 16; ++j)
      {
         training_data[i].attributes[j] = (training_data[i].attributes[j] - mean[j]) / dev[j]; 
	 test_data[i].attributes[j] = (test_data[i].attributes[j] - mean[j]) / dev[j];
      }
}

// Load the training and testing data from the file "letter-recognition.txt"
void load_data(instance training_data[], instance test_data[]) 
{
   ifstream input;
   char letter;

   input.open("letter-recognition.txt");

   // The first half of the instances in the file are loaded as the testing set 
   for (int i = 0; i < 10000; ++i)
   {
      input >> letter;
      input.ignore();
      test_data[i].letter = letter - 65;
      for (int j = 0; j < 16; ++j)
      {
	 input >> test_data[i].attributes[j]; 
	 input.ignore();
      }
   }

   // The second half of the file instances are loaded as the training set
   for (int i = 0; i < 10000; ++i)
   {
      input >> letter;
      input.ignore();
      training_data[i].letter = letter - 65;;
      for (int j = 0; j < 16; ++j)
      {
         input >> training_data[i].attributes[j];
	 input.ignore();
      }
   }
   input.close();
}
