

class Perceptron(object):
	def __init__(self, input_num,activator):
		super(Perceptron, self).__init__()
		self.weights = [0.0 for _ in range(input_num)]
		self.activator = activator
		self.bias = 0.0
	def __str__(self):
		return 'weights\t:%s\nbias\t:%f\n' %(self.weights,self.bias)

	def predict( self, input_vec):
		t = map( lambda (x,w):x*w,zip(input_vec,self.weights))
		return self.activator(sum([x*w for (x,w) in zip(input_vec, self.weights)])+self.bias)

	def train(self, input_vec, labels, iteration, rate):
		for i in range(iteration):
			self._one_iteration(input_vec,labels, rate)

	def _one_iteration(self, input_vecs, labels, rate):
		samples = zip(input_vecs, labels)
		for (input_vec, label) in samples:
			output = self.predict(input_vec)
			self._update_weights(input_vec, output, label, rate)

	def _update_weights(self, input_vec, output, label, rate):
		delta = label - output
		print "input_vec: ", input_vec,
		print " output: ",output, " label: ",label, 
		print " rate: ",rate, " weights: ",self.weights,
		print " bias: ",self.bias,
		print " delta: ",delta;
		self.weights = map(
			lambda tp:tp[1] + rate*delta * tp[0],
			zip(input_vec,self.weights))
		print(self.weights)
		self.bias += rate * delta		
		print(self.bias)
		print("====")
def f(x):
	return 1 if x > 0 else 0
	
	
def get_training_dataset():
	input_vecs = [[1,1], [0,0], [1,0], [0,1]]
	labels = [0, 0, 1, 1]
	return input_vecs, labels	

def train_and_perceptron():
	p = Perceptron(2, f)
	input_vecs, labels = get_training_dataset()
	p.train(input_vecs, labels, 10, 0.1)
	return p

if __name__ == '__main__': 
	and_perception = train_and_perceptron()
	print and_perception
	print '1 and 1 = %d' % and_perception.predict([1, 1])
	print '0 and 0 = %d' % and_perception.predict([0, 0])
	print '1 and 0 = %d' % and_perception.predict([1, 0])
	print '0 and 1 = %d' % and_perception.predict([0, 1])