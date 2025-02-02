import numpy as np
import matplotlib.pyplot as plt
#data
input_size=2
hidden_size=3
output_size=1
learning_rate=0.01
epochs=1000
#initialize
weights_input_hidden = np.random.rand(input_size,hidden_size)
biases_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.rand(hidden_size,output_size)
biases_output = np.zeros((1,output_size))


#activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Mean squared error loss
def mse_loss(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

#train data
X=np.array([[0,0],[0,1],[1,0],[1,1]])

Y=np.array([[0],[1],[1],[0]])

#model
loses = []
for epoch in range(epochs):
    z1=np.dot(X, weights_input_hidden) + biases_hidden
    a1=sigmoid(z1)

    z2=np.dot(a1, weights_hidden_output) + biases_output
    predicted_output=sigmoid(z2)
    loss=mse_loss(Y,predicted_output)
    loses.append(loss)

    #...backward pass
    output_error = Y - predicted_output
    output_delta = output_error*sigmoid(predicted_output) * (1 - sigmoid(predicted_output))

    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error*sigmoid(z1)*(1 - sigmoid(z1))

    # update weights and biases
    weights_hidden_output+=z1.T.dot(output_delta) * learning_rate
    weights_input_hidden+=X.T.dot(hidden_layer_delta) * learning_rate
    biases_output=np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    biases_input=np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate



#plot the loss curve
plt.plot(range(epochs), loses)
plt.xlabel('Epochs')
plt.ylabel('mse')
plt.title("hbbdbcjb")
plt.show()

# Make predictions after training
test_data = np.array(([0,0], [0,1], [1,0], [1,1]))
predictions = sigmoid(np.dot(sigmoid(np.dot(test_data,'w1') + 'b1')))
print("predictions after training:")
print(predictions)

