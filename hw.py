import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

x_train_full = x_train_full.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

x_train_full = x_train_full.reshape((-1, 28*28))  # Flatten
x_test = x_test.reshape((-1, 28*28))

x_train, y_train = x_train_full[:100], y_train_full[:100]
x_val, y_val = x_train_full[100:200], y_train_full[100:200]
x_test, y_test = x_test[:100], y_test[:100]
# x_train, y_train = x_train_full[:10000], y_train_full[:10000]
# x_val, y_val = x_train_full[10000:20000], y_train_full[10000:20000]
# x_test, y_test = x_test[:10000], y_test[:10000]

y_train_vecs = to_categorical(y_train, 10)
y_val_vecs = to_categorical(y_val, 10)
y_test_vecs = to_categorical(y_test, 10)

image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
    [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
    [0, 0, 0, 0, 0, 255, 255, 0, 0, 0],
    [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
    [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
    [0, 0, 255, 255, 0, 0, 0, 0, 0, 0],
    [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
    [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.float32)

# normalize to 0, 1
image = image / 255

# 100 x 1 vector for input to the neural net
# x = image.flatten()

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

# W1 = np.zeros((10, 784))  # takes image, 10 perceptrons
# b1 = np.zeros(10)

# W2 = np.zeros((10, 10))   # 10 perceptrons
# b2 = np.zeros(10)

# W3 = np.zeros((10, 10))   # 10 perceptrons
# b3 = np.zeros(10)

# W1 = np.random.randn(10, 784) * np.sqrt(2 / 784)
# W2 = np.random.randn(10, 10) * np.sqrt(2 / 10)
# W3 = np.random.randn(10, 10) * np.sqrt(2 / 10)

# W1 = np.random.randn(10, 784)
# W2 = np.random.randn(10, 10)
# W3 = np.random.randn(10, 10)

b1 = np.zeros(10)
b2 = np.zeros(10)
b3 = np.zeros(10)

W1 = np.random.rand(10, 784) - 0.5
W2 = np.random.rand(10, 10) - 0.5
W3 = np.random.rand(10, 10) - 0.5

learning_rate = 0.01
epochs = 100
# y_true = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32) # encode 2 as target

# MSE was TERRIBLE
# def mse(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)

# i found people doing this
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def cross_entropy(y_true, y_pred):
    eps = 1e-9
    return -np.sum(y_true * np.log(y_pred + eps)) / y_true.shape[0]


for epoch in range(epochs):
    tot_loss = 0
    for x, y_true in zip(x_train, y_train_vecs):
        
    
        # forward pass 
        z1 = W1 @ x + b1
        a1 = relu(z1)

        z2 = W2 @ a1 + b2
        a2 = relu(z2)

        z3 = W3 @ a2 + b3
        output = softmax(z3)
        
        # loss = mse(y_true, output)
        loss = cross_entropy(y_true, output)
        
        # abuse chain rule for backprop
        dL_dz3 = (output - y_true)
        dL_dW3 = np.outer(dL_dz3, a2) # outer bcus it gives the right shape
        dL_db3 = dL_dz3  # dz3/zb3 = 1 :)
        dz3_da2 = W3.T # this gradient is just the transpose of W3
        dL_da2 = dL_dz3 @ dz3_da2
        dL_dz2 = dL_da2 * relu_derivative(z2)
        dL_dW2 = np.outer(dL_dz2, a1)
        dL_db2 = dL_dz2  # dz2/zb2 = 1 :)
        dz2_da1 = W2.T 
        dL_da1 = dL_dz2 @ dz2_da1
        dL_dz1 = dL_da1 * relu_derivative(z1)
        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1  # dz1/zb1 = 1 :)

        # update weights and biases
        W1 -= learning_rate * dL_dW1
        b1 -= learning_rate * dL_db1
        W2 -= learning_rate * dL_dW2
        b2 -= learning_rate * dL_db2
        W3 -= learning_rate * dL_dW3
        b3 -= learning_rate * dL_db3
        tot_loss += cross_entropy(y_true, output)
        
    avg_loss = tot_loss / len(x_train)
    
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")


# # run image through the trained network
# if __name__ == '__main__':
#     print("Final Weights and Biases:")
#     print("W1:", W1)
#     print("b1:", b1)
#     print("W2:", W2)
#     print("b2:", b2)
#     print("W3:", W3)
#     print("b3:", b3)
#     z1 = W1 @ x + b1
#     a1 = relu(z1)
#     z2 = W2 @ a1 + b2
#     a2 = relu(z2)
#     z3 = W3 @ a2 + b3
#     output = z3
#     print("Output after training:", output)
#     print("Predicted class:", np.argmax(output))
#     # accuracy calc
#     accuracy = np.mean(np.argmax(output) == np.argmax(y_true))
#     print("Accuracy:", accuracy)
#     print("Expected class:", np.argmax(y_true))

correct = 0
for x, y_true in zip(x_test, y_test_vecs):
    z1 = W1 @ x + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    a2 = relu(z2)
    z3 = W3 @ a2 + b3
    output = z3
    if np.argmax(output) == np.argmax(y_true):
        correct += 1

accuracy = correct / len(x_test)
print("Test Accuracy:", accuracy)
