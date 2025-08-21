import numpy as np
# data _ input features
A= [[1,0,1], # row 1
     [0,1,1], # row 2
     [0,0,1], # row 3 and so on
     [1,1,1],
     [0,1,1],
     [1,0,1]
     ]
# target features
B = [0,1,0,1,1,0]
X = np.array(A)
y = np.array(B)
# for each feature in a row we are going to have our input neuron
# which are going to multiply each feature with a weight add bais and apply activation function to the output
weights = np.array([0.5,0.48,-0.7])
# changes start
def relu2deriv(output):
    return output>0
#changes end
# "alpha" is the learning rate
alpha = 0.2
hidden_size = 4
np.random.seed(1)
y = y.T
def relu(x):
    return (x> 0) *x

# weights are to be of size features/inputs * no of neurons in this layer
# weights_0_1 are weights of first hidden layer having inputs 3(features) * hidden_neurons 4
weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
# weights_1_2 are weights of output layer that has 4 inputs from 4 neurons (1 from each)
# and has one neuron
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1
# changes start
layer_0 = X[0]
for iteration in range(60):
    layer_2_error = 0
    for i in range(len(X)):
        layer_0 = X[i:i + 1] # [[1,0,1]] if you write just X[i] it would be a [1,0,1]
        print(layer_0)
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)
        layer_2_error += np.sum((layer_2 - y[i:i + 1]) ** 2)
        layer_2_delta = (y[i:i + 1] - layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
        #update weights between hidden layer and output layer.
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        #pdate weights between hidden layer and input layer.
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
    if (iteration % 10 == 9):
        print("Error:" + str(layer_2_error))
#changes end

'''

why the shape is (3,hidden)?
winputneuron
weights_0_1 = [
 [w11, w12, w13, w14],      # from input 1 to each hidden neuron
 [w21, w22, w23, w24],      # from input 2 to each hidden neuron
 [w31, w32, w33, w34]       # from input 3 to each hidden neuron
]  # shape (3,4)

so neuron1 has weights w11,w21,w31
so neuron2 has weights w12,w22,w32
so neuron3 has weights w13,w23,w33
so neuron3 has weights w14,w24,w34

when we have input = [x1, x2, x3]        # shape (3,) its to be sent to every neuron
at every neuron a dot is going to take place 
[x1,x2,x3] dot [ [w11, w12, w13, w14],   = Ans]   
                 [w21, w22, w23, w24],     
                 [w31, w32, w33, w34]       
                ]  
i.e (1,3) dot (3,4) = (1,4) = Ans
[ans1,ans2,ans3,ans4] dot [ [w1]
                            [w2],
                            [w3],
                            [w4]]
i.e (1,4) dot (4,1) = 1,1
#---------------------------------------------------------------
np.random.random((3, hidden_size))
Creates a matrix of shape (3, hidden_size) where each value is a random number in [0, 1).
3 → number of input features (neurons in the input layer)
hidden_size → number of neurons in the hidden layer
Multiply by 2: turns the range [0, 1) → [0, 2).
Subtract 1: shifts the range [0, 2) → [–1, 1).


'''
"""
How does stochastic gradient descent work? As you saw in the previous example, it
performs a prediction and weight update for each training example separately. In other
words, it takes the first streetlight, tries to predict it, calculates the weight_delta, and
updates the weights. Then it moves on to the second streetlight, and so on. It iterates
through the entire dataset many times until it can find a weight configuration that works
well for all the training examples

(Full) gradient descent updates weights one dataset at a time.
 As introduced in chapter 4, another method for learning an entire dataset is gradient
descent (or average/full gradient descent). Instead of updating the weights once for each
training example, the network calculates the average weight_delta over the entire dataset,
changing the weights only each time it computes a full average.

Batch gradient descent updates weights after n examples.
 This will be covered in more detail later, but there’s also a third configuration that sort
of splits the difference between stochastic gradient descent and full gradient descent.
Instead of updating the weights after just one example or after the entire dataset of
examples, you choose a batch size (typically between 8 and 256) of examples, after
which the weights are updated.

in the process of gradient descent, each
training example asserts either up pressure or down pressure on the weights. On average,
there was more up pressure for the middle weight and more down pressure for the other
weights. Where does the pressure come from? Why is it different for different weights?

layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
Takes the error at the output layer (layer_2_delta),
Multiplies by the transpose of the weights connecting layer 1 → layer 2 to propagate error backward,
Multiplies element-wise by the derivative of ReLU to keep only the parts that affect the output.

weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
Updates weights between layer 1 → layer 2,
Using the learning rate alpha,
And the gradient: activations from layer 1 (layer_1.T) × error at layer 2 (layer_2_delta).
weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
Same idea, but for weights between layer 0 → layer 1,s
Using the input activations (layer_0.T) × backpropagated error at layer 1 (layer_1_delta).
"""