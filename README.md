
# Features

-  Implements a simple yet effective feedforward neural network with configurable input, hidden, and output layers.
-  Utilizes pthreads to parallelize neural network training for improved performance.

-  TimeSeriesNN uses a feedforward neural network with:

- Configurable number of input features (default: 5).
- Tanh activation in the hidden layer.
- Linear activation in the output layer.
- Stochastic gradient descent for optimization.
- Mean squared error as the loss function.
     
![AI](https://github.com/user-attachments/assets/54d803c2-5d52-4402-805f-de6e31c63be4)

# Requirements

- GCC or compatible C compiler
- pthread library
- math library

# Configuration

- The following parameters can be adjusted in the source code:

      #define MAX_SAMPLES 1000        
      #define WINDOW_SIZE 300         
      #define NUM_FEATURES 5          
      #define HIDDEN_LAYER_SIZE 10    
      #define OUTPUT_SIZE 1          
      #define LEARNING_RATE 0.01      
      #define PREDICTION_HORIZON 50  
      #define MAX_THREADS 4           
