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
alpha = 0.1

for iteration in range(40):
    total_error_for_whole_dataset = 0
    for row in range(len(y)):
        input = X[row, :]
        prediction_goal = y[row]
        prediction = input.dot(weights)  # scalar
        error = (prediction_goal - prediction) ** 2
        total_error_for_whole_dataset += error
        delta = prediction_goal - prediction
        weights = weights - (alpha * (input * delta))
        print("Prediction : " , str(prediction))

    print("total_error_for_whole_dataset : " , str(total_error_for_whole_dataset))

