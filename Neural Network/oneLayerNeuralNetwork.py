from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init__(self):
		# Seed the random number generator, so it genereates the same
		# numbers everytime the program runs
		random.seed(1)

		# Model a single neuron, with 3 input connections and 1 output connection
		# assign random weights to 3x1 matrix with balues in range -1 to 1 and
		# mean of 0
		self.synaptic_weights = 2* random.random((3,1)) - 1

	# Sigmoid function, which describes an s shaped curve
	# pass the weighted sum of the inputs through it and converts them to a
	# probability between 0 and 1
	def __sigmoid(self, x):
		return 1/(1 + exp(-x))


	# Calculates the derivative of the sigmoid function for gradient descent
	def __sigmoid_derivative(self, x):
		return x * (1-x)

	# To get the weighted sum of inputs compute dot prodcut of inputs and weights
	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):
			# pass the training set through neural net
			output = self.predict(training_set_inputs)

			# calculate error (desired - predicted)
			error = training_set_outputs - output

			# Gradient descent -> derivative of sigmoid gives us gradient/slope
			'''	we want to minimize error as we train
				do this by iteratively update weights
				calculate necessary adjustments by computing dot
				product of inputs transposed and the error multiplied
				by the gradient of the sigmoid curve '''
			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

			#adjust the weights
			self.synaptic_weights += adjustment


	def predict(self, inputs):
		# pass inputs through neural net
		return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == '__main__':

	#initialize single neuron neural network
	neural_network = NeuralNetwork()

	print 'Random starting synaptic weights:'
	print neural_network.synaptic_weights

	# The training set. 4 examples, each consisting of 3 input values
	# and 1 output value (Ax = b)
	# A
	training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
	# x
	training_set_outputs = array([[0,1,1,0]]).T

	# train the neural network using a training set
	# do it 10000 times and make small adjustments each time
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print 'New synaptic weights after training:'
	print neural_network.synaptic_weights

	# Test the neural network
	print 'Considering new situation [1, 0, 0] -> ?:'
	print neural_network.predict(array([1, 0, 0]))