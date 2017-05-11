from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys, getopt

from tensorflow.examples.tutorials.mnist import input_data #Para la descarga de datos automatica
import tensorflow as tf

FLAGS = None

class Graph:
	def __init__(self, network):
		self.net = network
	#	logits = mnist.inference(self.net.x,FLAGS.hidden1,FLAGS.hidden2)



class Network:
	def __init__(self, input, output,hidden):
		#for _ in range(hidden):
		#	self.W[_] = tf.Variable(tf.zeros([input, output])) #Futuro Constructor podrá crear múltiples capas hidden
		#Input
		self.x = tf.placeholder(tf.float32, [None, input])
		#Hidden
		self.Wh = tf.Variable(tf.random_normal([input, hidden]))
		self.bh = tf.Variable(tf.random_normal([hidden]))
		
		self.h = tf.nn.sigmoid(tf.matmul(self.x, self.Wh) + self.bh)
		#Output
		self.Wy = tf.Variable(tf.random_normal([hidden, output]))
		self.by = tf.Variable(tf.random_normal([output]))
		self.y_ = tf.placeholder(tf.float32, [None, output])
		
		self.y = tf.matmul(self.h, self.Wy) + self.by
		#self.y = tf.nn.softmax(tf.matmul(self.h, self.Wy) + self.by)
		
		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y)) #Calculo del error usando softmax, cross entropy y la media.
		#self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
		
		self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1)) #Se compara la salida activada y la que deberia ser y devuelve una lista de booleans [true,false,false,true]
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32)) #Se hace cast a una lista binarias Ej: [1,0,0,1] y se cualcula la media 1+0+0+1/4 = 0.5

		self.sess = tf.InteractiveSession()
		tf.global_variables_initializer().run() #iniciando variables y sesion
		
		
	def train(self, v_train,iterations=1000, lr=0.01,batch_size=100,evaluate=False,target=2):

		train_step = tf.train.GradientDescentOptimizer(lr).minimize(self.cross_entropy)
		accuracy_result=0
		i=0
		while i<iterations and accuracy_result<target:
			i+=1
			batch_xs, batch_ys = v_train.next_batch(batch_size)
			step,accuracy_result =self.sess.run([train_step,self.accuracy], feed_dict={self.x: batch_xs, self.y_: batch_ys}) #Dos maneras de escribir lo mismo
			#train_step.run(feed_dict={self.x: batch_xs, self.y_: batch_ys})
			if evaluate:
				print('Precision: ',accuracy_result)
			
	def test(self,test_xs,test_ys,evaluate=0):
		if evaluate==1:
			#loss = tf.reduce_sum(tf.square(self.y_ - self.y)) 
			loss = tf.reduce_sum(tf.square(self.y_ - tf.nn.softmax(self.y)))
			return self.sess.run(loss, feed_dict={self.x: test_xs,self.y_: test_ys})/len(test_xs)
			

		return self.sess.run(self.accuracy, feed_dict={self.x: test_xs,self.y_: test_ys}) #lanzamos el modelo con los valores de placeholder para accuracy
	
def main(argv):
	try:
		opts, args = getopt.getopt(sys.argv[1:],"ami:o:h:l:n:b:",["input=","output="])
	except getopt.GetoptError:
		print('tensordef.py -i <inputs> -o <outputs> -h <hidden> -l <learning rate> -n <iterations> -b <batch size> -m {monitor}')
		sys.exit(2)
	inputs=784
	outputs=10
	hidden=100
	learning=0.5
	iterations = 5500
	batch_size=200
	m=False
	for opt, arg in opts:
		if opt == '-a':
			print('tensordef.py -i <inputs> -o <outputs> -h <hidden> -l <learning rate> -n <iterations> -b <batch size> -m {monitor}')
			sys.exit()
		elif opt in ("-i", "--input"):
			inputs = arg
		elif opt in ("-o", "--output"):
			outputs= arg
		elif opt in ("-h"):
			hidden= arg
		elif opt in ("-l"):
			learning= arg
		elif opt in ("-n"):
			iterations= arg
		elif opt in ("-b"):
			batch_size= arg
		elif opt == ("-m"):
			m= True
		 
	mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)#descarga de datos automatica en /tmp
	net= Network(inputs,outputs,hidden)
	net.train(mnist.train, iterations,learning,batch_size,m)#mnist.train tiene 55000 images(784px) and labels(10)
	test_result = net.test(mnist.test.images, mnist.test.labels)
	print('Porcentaje de acierto Test: ' , round(test_result*100,2) , '%') #mnist.test.images [10000,784] mnist.test.labels [10000,10] 
	
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)