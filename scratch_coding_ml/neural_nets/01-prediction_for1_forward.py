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
input = X[0,:]
prediction_goal = y[0]

for iteration in range(20):
     prediction = input.dot(weights)  # scalar
     error = (prediction_goal - prediction) ** 2
     delta = prediction_goal - prediction
     weights = weights - (alpha * (input * delta))
     print(weights)
     print("Error : " , str(error) +" - " ," Prediction : "+ str(prediction) )


'''
2️⃣ X.dot(weights) → vector
When you multiply the whole matrix X with weights:
Example calculation:

Row 1: 1∗0.5+0∗0.48+1∗(−0.7)=0.5+0−0.7=−0.2
Row 2: 0∗0.5+1∗0.48+1∗(−0.7)=0+0.48−0.7=−0.22
Row 3: 0∗0.5+0∗0.48+1∗(−0.7)=−0.7
…and so on for all 6 rows.

So:
X.dot(weights)
# → array([-0.2 , -0.22, -0.7 ,  0.28, -0.22, -0.2 ])
That’s 6 predictions — one per row of X.

3️⃣ np.dot(input, weights) → scalar
Here input is just the first row:
1∗0.5+0∗0.48+1∗(−0.7)=0.5+0−0.7=−0.2
np.dot(input, weights) → -0.2
That’s only one prediction — for the first row.
'''