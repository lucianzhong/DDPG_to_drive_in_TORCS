import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

# actor包含online policy和 target policy 两张神经网络， 其结构是一样的
#代码中使用了2个隐藏层，分别有300和600个隐藏单元。 输出包括3个连续动作，转向Steering，
#这是一个具有tanh激活功能的单个单位（其中-1表示最右转弯，+1表示最大左转弯）。 
#加速度Acceleration,，是具有S形激活功能的单个单元（其中0表示无气体，1表示全气体）。 制动Brake，另一个具有S形激活功能的单元（其中0表示无制动，1表示制动）

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)   #使用Adam作为gradient descent的算法
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run( self.optimize, feed_dict={self.state: states, self.action_gradient: action_grads} )

    """
	    Target Network
	It is a well-known fact that directly implementing Q-learning with neural networks proved to be unstable in many environments including TORCS. 
	Deepmind team came up the solution to the problem is to use a target network, where we created a copy of the actor and critic networks respectively, 
	that are used for calculating the target values. 
	The weights of these target networks are then updated by having them slowly track the learned networks:
		θ′←τθ+(1−τ)θ′
	where τ≪1. This means that the target values are constrained to change slowly, greatly improving the stability of learning.
    """
    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("create_actor_network")
        S = Input(shape=[state_size])   
        # used 2 hidden layers with 300 and 600 hidden units respectively
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        #为了限定policy网络的输出action范围，使用tanh对steer，sigmoid对accelerate和brake，作为bound函数，进行范围限定
        #The output consist of 3 continuous actions, Steering, which is a single unit with tanh activation function (where -1 means max right turn and +1 means max left turn). 
        #Acceleration, which is a single unit with sigmoid activation function (where 0 means no gas, 1 means full gas). 
        # Brake, another single unit with sigmoid activation function (where 0 means no brake, 1 bull brake)
        Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)  
        Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
        Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1) 
        #V = merge([Steering,Acceleration,Brake],mode='concat')          
        V = Dense(3,activation='tanh')(h1) 
        model = Model(input=S,output=V)
        return model, model.trainable_weights, S

        """
        代码使用了名为Merge层将3个输出组合在一起（在keras2.2.0的版本中Merge已经取消）。聪明的读者可能会问为什么不使用像这样的传统密集功能
		V = Dense(3,activation='tanh')(h1)
		这是有原因的。首先使用3个不同的Dense（）函数允许每个连续动作具有不同的激活功能，例如，使用tanh（）进行加速没有意义，因为tanh在[-1,1]范围内，而加速度在范围内[0,1]
		还请注意，在最后一层我们使用了正常的初始化 μ = 0，σ = 1e-4，以确保policy的初始产出接近于零。
		
		"""

