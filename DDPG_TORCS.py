import numpy as np
import random
import argparse
import json
import tensorflow as tf
import timeit

# for the pre-trained model
"""
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.engine.training import collect_trainable_weights
"""
# self-defined modules
from gym_torcs_DDPG import TorcsEnv  # specifiy the env
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU

OU = OU()       # 使用Uhlenbeck-Ornstein随机过程（下面简称UO过程），作为引入的随机噪声

def playGame(train_indicator= 1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    np.random.seed(1337) #seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
    # parameters for iteration
    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 30   # steps for an episode
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #from keras import backend as K
    #K.set_session(sess)

    # the actor & critic networks
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=False, throttle=True,gear_change=False)
    #print("env.action_space")

    print("TORCS Experiment Start      ")
    # for each episode:
    # for each time-step:
    # actor choose actions, environemnt do the actions 
    for i in range(episode_count):
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) # the states
        total_reward = 0

        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))  #keras Model API
            # print("s_t.shape[0]",s_t.shape[0])  #29
            # print("a_t_original",a_t_original.shape)  # (1,3)
            #如何在连续域中设计正确的探索算法。 在Q-learing我们使用了ε贪婪策略，其中代理在某个百分比的时间内尝试随机动作。 
            #然而，这种方法在TORCS中不能很好地工作，因为我们有3个动作[转向，加速，制动]。 如果我只是从均匀随机分布中随机选择动作，
            #我们将生成一些无聊的组合[例如：制动的值大于加速度的值而车辆根本不移动）。 因此，我们使用Ornstein-Uhlenbeck过程添加噪声来进行探索。
            #Ornstein-Uhlenbeck过程
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)
            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            # print(a_t[0]) #[ 0.00066324  0.42420975 -0.09446412]

            ob, r_t, done, info = env.step(a_t[0])# the action is a_t[0] and get the observations from TORCS
            s_t1 = np.hstack( (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm) )
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            #When training the network, random mini-batches from the replay memory are used instead of most the recent transition, which will greatly improve the stability. 
            #Do the batch update #Experience Replay
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])  

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
            # training
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t)  # keras API
                #而确定性策略则决定简单点，虽然在同一个状态处，采用的动作概率不同，但是最大概率只有一个，如果我们只取最大概率的动作，去掉这个概率分布，那么就简单多了。
                #即作为确定性策略，相同的策略，在同一个状态处，动作是唯一确定的，即策略变成 pi{theta}(s) = a
                a_for_grad = actor.model.predict(states)  #然后使用采样的策略梯度更新actor策略, 召回a是确定性政策
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1        
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
            step += 1
            if done:
                break

        # the saving process
        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)
                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
