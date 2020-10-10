# -*- coding: utf-8 -*-

"""
Names: Bar Dalal and Elay Ben David
Class: YB4

The model, based on an ANN, predicts someone's genter given features about him, such as his weight and height.
The data is given so that:
the constant 135 is substracted from the actual weight (lb), and 66 is substracted from the actual height (in).
In regard to the predictions: male is represented by 0 and female is represented by 1.
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
  """
  Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  The function gets a number and activates the sigmoid function on that number
  The function returns this value
  """
  return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
  """
  Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  The function gets a number and activate the sigomid function derivative on that number
  The function returns this value
  """
  return x * (1 - x)


def mse_loss(y_true, y_pred):
  """
  The function gets a matrix that is the target outputs and a matrix that is the actual predictions of the model
  The function returns the total error of the model, using the mean squared error loss function
  """
  return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:
  
  """
  A neural network with:
    - some inputs
    - a hidden layer with some neurons
    - an output layer with 1 neuron
  """
  
  def __init__(self, num_inputs, num_hidden):
    """
    Constructor of the neural network.
    The function gets the number of neurons in the input layer and the number of neurons in the hidden layer
    The function create an object of the class "OurNeuralNetwork"
    """
    # Weights: 
    self.weights_hidden= np.random.rand(num_inputs, num_hidden) # matrix of the weights connecting the inputs to the hidden layer
    self.weights_output= np.random.rand(num_hidden, 1) # matrix of the weights connecting the hidden layer to the output layer
    # Biases:
    self.biases_hidden= np.random.rand(num_hidden, 1) # matrix of the biases of the hidden layer
    self.bias_output= np.random.rand(1, 1) # 1X1 matrix of the bias of the output layer
    self.error_history= [] # a list containing the total error of each epoch
    self.epoch_list= [] # a list containing the numbers of the total epochs occured
    
    
  def train (self, data, all_y_trues, learning_rate= 0.1, epochs= 5000):
    """
    The function gets a matrix of the data and a matrix of the target output.
    If the function doesn't get a learning rate and a number of epochs, 
    then the values of these parameters will be set to 0.1 and 5000 respectively.
    The function trains the model to give the most accurate predictions.
    """
    for epoch in range(epochs):
        # feedforward through the network:
        # hidden layer:
        hidden_layer_inputs= np.dot(data, self.weights_hidden) + self.biases_hidden.T
        hidden_layer_outputs= sigmoid(hidden_layer_inputs)
        # output layer:
        output_layer_inputs= np.dot(hidden_layer_outputs, self.weights_output) + self.bias_output
        output_layer_outputs= sigmoid(output_layer_inputs)
        
        # calculating the error and backpropagation: 
        # output layer:    
        output_layer_error= output_layer_outputs - all_y_trues 
        output_layer_delta= output_layer_error * deriv_sigmoid(output_layer_outputs)
        # hidden layer:
        hidden_layer_error= np.dot(output_layer_delta, self.weights_output.T)
        hidden_layer_delta= hidden_layer_error * deriv_sigmoid(hidden_layer_outputs)
        # updating the weights and the biases:
        # output layer:    
        self.weights_output-= learning_rate * np.dot(hidden_layer_outputs.T, output_layer_delta)
        self.bias_output-= learning_rate * np.sum(output_layer_delta)
        # hidden layer:
        self.weights_hidden-= learning_rate * np.dot(data.T, hidden_layer_delta) 
        self.biases_hidden-= learning_rate * np.sum(hidden_layer_delta)
        
        self.update_lists(all_y_trues, output_layer_outputs, epoch) # to create a graph
        


  def update_lists(self, all_y_trues, output_layer_outputs, epoch):
      """
      The function gets a matrix of the target outputs, a matrix of the actual predictions and the number of the current epoch  
      The function updates the lists containing the total errors and the number of epochs
      The functions prints the total error every 10 epochs
      """
      self.error_history.append(mse_loss(all_y_trues, output_layer_outputs))
      self.epoch_list.append(epoch)
      if epoch % 10 == 0:
          print("Epoch %d loss: %.3f" % (epoch, mse_loss(all_y_trues, output_layer_outputs)))
      
  
  def predict(self, new_inputs):
    """
    The function gets a matrix of a new data to test the model
    The function returns the predicted genter corresponding the parameter
    """
    # feedforward:
    h_output= sigmoid(np.dot(new_inputs, self.weights_hidden) + self.biases_hidden.T) # the hidden layer output
    return sigmoid(np.dot(h_output, self.weights_output) + self.bias_output)
      

def main():
    # Define dataset:
    data = np.array([   
        [-2, -1],  # Alice
        [25, 6],   # Bob
        [17, 4],   # Charlie
        [-15, -6], # Diana
        ])
    all_y_trues = np.array([
        [1], # Alice
        [0], # Bob
        [0], # Charlie
        [1], # Diana 
        ])
    
    network = OurNeuralNetwork(data.shape[1], 2) # our ANN
    network.train(data, all_y_trues) # train the neural network
    # Make some predictions:
    emily= np.array([-7, -3]) # 128 pounds, 63 inches- F
    frank= np.array([20, 2])  # 155 pounds, 68 inches- M
    danit= np.array([-21, -2])  # 114 pounds, 64 inches- F
    avi= np.array([28, 4]) # 163 pounds, 70 inches- M
    
    print('The predictions:')
    print("Emily: %.3f" % network.predict(emily)) 
    print("Frank: %.3f" % network.predict(frank)) 
    print("danit: %.3f" % network.predict(danit))
    print("avi: %.3f" % network.predict(avi))

    # displaying the graph
    plt.plot(network.epoch_list, network.error_history)
    plt.xlabel('Epoch')
    plt.ylabel('Total error')
    plt.show()


if __name__ == "__main__":
    main()
