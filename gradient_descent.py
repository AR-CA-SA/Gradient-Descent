import numpy as np
import matplotlib.pyplot as plt

"""
Gradient Descent is an algorithm use to find the 
minimum value of a function quickly, it is based on
the concept of the gradient of a scalar function which
naturally points to the direction of steepest increase. 
In the case of gradient descent, we care about the 
direction of steepest decrease.


Gradient descent is use to minimize the RMS Error 
of the predicted values and the actual values

Gradient descent finds the minimum of a convex function. However,
it might need adjustments for non-convex problems
(finding local min vs global min)
"""

#RMS 

def root_mean_square(predicted, actual):

    cost = np.sum(np.square(predicted - actual)/len(actual))
    return cost



def gradient_descent(independent_variable, actual_target,iterations = 1000 , learning_rate = 0.0001, stopping_threshold = 1e-6):
    current_weight = 0.1
    current_bias = 0.01
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(independent_variable))
    costs = []
    weights = []
    previous_cost = None
    for i in range(iterations):
        target_predicted = (current_weight * independent_variable) + current_bias
        current_cost = root_mean_square(target_predicted,actual_target)
        if previous_cost and abs(previous_cost - current_cost)<= stopping_threshold:
            break
        previous_cost = current_cost

        costs.append(current_cost)
        weights.append(current_weight)
        weight_derivative = -(2/n)*np.sum(independent_variable*(actual_target-target_predicted))
        bias_derivative = -(2/n)*np.sum((actual_target-target_predicted))
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)

    plt.figure(figsize=(8,6))
    plt.plot(weights,costs)
    plt.scatter(weights,costs)
    plt.show()
    return current_weight, current_bias

def main():

   # Data
   X = np.array([32.5, 53.4, 61.5, 47.4, 59.8, 
   55.1, 52.2, 39.2, 48.1, 52.5, 
   45.4, 54.3, 44.1, 58.1, 56.7, 
   48.9, 44.6, 60.2, 45.6, 38.8])
   Y = np.array([31.7, 68.7, 62.5, 71.5, 87.2,
   78.2, 79.6, 59.1, 75.3, 71.3,
   55.1, 82.4, 62.0, 75.3, 81.4, 
   60.7, 82.8, 97.3, 48.8, 56.8])

   # Estimating weight and bias using gradient descent
   estimated_weight, estimated_bias = gradient_descent(X, Y, iterations = 2000)
   print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {estimated_bias}")

   # Making predictions using estimated parameters
   Y_pred = estimated_weight*X 

   #Plotting the regression line
   plt.figure(figsize = (8,6))
   plt.scatter(X, Y, marker = 'o', color = 'red')
   plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color = 'blue')
   plt.ylabel("X")
   plt.xlabel("Y")
   plt.show()

if __name__ == "__main__":
    main()