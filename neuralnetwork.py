import numpy as np
import cv2

class NeuralNetwork:
	# NeuralNetwork initializaiton
	def __init__(self, inputs, outputs, alpha = 0.1):
		# input layer
		self.a_k = inputs
		# learning rate alpha
		self.alpha = alpha
		# number of nodes in hidden layer
		self.hidden_size = 300
		# weights from input layer to hidden layer
		self.weights_kj = np.random.randn(self.a_k.shape[1], self.hidden_size)
		# weights from hidden layer to output layer
		self.weights_ji = np.random.randn(self.hidden_size, 5)
		# expected output layer
		self.y = outputs
		# actual output layer
		self.a_i = np.full_like(self.y, 0)

	# feed forward function
	def feed_forward(self):
		# hidden layer
		self.a_j = self.sigmoid(np.dot(self.a_k, self.weights_kj))
		# output layer
		self.a_i = self.sigmoid(np.dot(self.a_j, self.weights_ji))

	# calculate delta_i
	def delta_i(self):
		# error value
		err = self.y - self.a_i
		in_i = self.a_i
		# matrix filled with 1's
		ones = np.full_like(in_i, 1.0)
		# derivative sigmoid value for a_i
		derivative_sigmoid_i = np.multiply(self.sigmoid(in_i), ones - self.sigmoid(in_i))
		delta_i = np.multiply(err, derivative_sigmoid_i)

		return delta_i

	# calculate delta_j
	def delta_j(self, delta_i):
		in_j = self.a_j
		# matrix filled with 1's
		ones = np.full_like(in_j, 1.0)
		# derivative sigmoid value for a_j
		derivative_sigmoid_j = np.multiply(self.sigmoid(in_j), ones - self.sigmoid(in_j))
		# part of the delta_j equation
		weights_ji_dot_delta_i = np.dot(self.weights_ji, np.transpose(delta_i))
		delta_j = np.multiply(derivative_sigmoid_j, np.transpose(weights_ji_dot_delta_i))

		return delta_j

	# back propagation
	def back_prop(self):
		# update weights between output layer and hidden layer
		delta_i = self.delta_i()
		# alpha * a_j
		alphaxa_j = np.multiply(np.full_like(self.a_j, self.alpha), self.a_j)
		# update weights_ji
		self.weights_ji = self.weights_ji + np.matmul(np.transpose(alphaxa_j), delta_i)

		# update weights between hidden layer and input layer
		delta_j = self.delta_j(delta_i)
		# alpha * a_k
		alphaxa_k = np.multiply(np.full_like(self.a_k, self.alpha), self.a_k)
		# update weights_kj
		self.weights_kj = self.weights_kj + np.matmul(np.transpose(alphaxa_k), delta_j)

	# sigmoid function for array
	def sigmoid(self, A):
		B = []
		for x in A:
			B.append(1.0 / (1.0 + np.exp(-x)))
		return np.array(B)

# train the neural network
def train(inputs, outputs, alpha, iterations):
	# create NeuralNetwork class
	nn = NeuralNetwork(inputs, outputs, alpha)
	# train for number of iterations
	for i in range(iterations):
		# print which iteration currently running
		print('iteration: ' + str(i), end='\r')
		nn.feed_forward()
		nn.back_prop()

	return nn

# calculate the accuracy
def correctness(Answer, Guess):
	# set up confusion matrix
	Confusion = np.zeros((5,5))
	for i in range(Answer.shape[0]):
		# expected digit
		answer = np.nonzero(Answer[i])[0][0]
		# pick largest number if more than one 1 in the output array
		if np.sum(Guess[i]) > 1:
			guess = round(max(np.nonzero(Guess[i])[0]))
		# print if all 0's
		elif np.sum(Guess[i]) == 0:
			guess = 0
		# actual digit
		else:
			guess = round(np.nonzero(Guess[i])[0][0])

		# update confusion matrix
		Confusion[answer][guess] += 1
		num_correct = np.sum(np.diagonal(Confusion))

	return str((num_correct / np.sum(Confusion)) * 100) + '% correct.', Confusion

if __name__ == "__main__":

	# get training images as input in an array
	inputs = []
	for i in range(28038):
		filename = 'train_img/train_image_' + str(i) + '.bmp'
		img = cv2.imread(filename, 0)
		inputs.append([item for sublist in img for item in sublist])

	inputs = np.array(inputs)

	# get expected traning output in an array
	outputs = []
	with open('train_labels.txt', 'r') as openfile:
		for line in openfile:
			outputs.append([int(n) for n in line.split(' ')])
	outputs = np.array(outputs)
	
	# get testing images as input in an array
	test_inputs = []
	for i in range(2561):
		filename = 'test_img/test_image_' + str(i) + '.bmp'
		img = cv2.imread(filename, 0)
		test_inputs.append([item for sublist in img for item in sublist])

	test_inputs = np.array(test_inputs)

	# get expected testing output in an array
	test_outputs = []
	with open('test_labels.txt', 'r') as openfile:
		for line in openfile:
			test_outputs.append([int(n) for n in line.split(' ')])
	test_outputs = np.array(test_outputs)

	# # TESTING ALPHA VALUE
	# for x in range(1, 10):
	# 	iterations = 100
	# 	alpha = x / 10
	# 	nn = train(inputs, outputs, alpha, iterations)
	# 	print('alpha: ' + str(alpha) + ' ----- ' + correctness(outputs, nn.a_i))

	# TRAIN
	iterations = 50
	alpha = 0.1
	nn = train(inputs, outputs, alpha, iterations)

	# TEST
	nn.a_k = test_inputs
	nn.y = test_outputs
	nn.feed_forward()

	# RESULTS
	accuracy = correctness(test_outputs, nn.a_i)
	print(accuracy[0])
	print(accuracy[1])

	# # MULTIPLE RUNS OF TRAINING THEN A TEST
	# for x in range(100):
	# 	# TRAIN
	# 	iterations = 50
	# 	alpha = 0.1
	# 	nn = train(inputs, outputs, alpha, iterations)

	# 	# TEST
	# 	nn.a_k = test_inputs
	# 	nn.y = test_outputs
	# 	nn.feed_forward()

	# 	# RESULTS
	# 	accuracy = correctness(test_outputs, nn.a_i)
	# 	print(accuracy[0])
	# 	print(accuracy[1])