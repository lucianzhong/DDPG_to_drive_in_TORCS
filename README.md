Requirements:    
---------
1. We are assuming you are using Ubuntu 14.04 LTS/16.04 LTS machine and installed,version do matters      
2. Python 3    
3. Keras 1.1.0   
4. Tensorflow r0.10   
5. gym_torcs       

How to install?
--------
https://blog.csdn.net/ss910/article/details/77618425    
(need to use sudo when run python snakeoil3_gym.py)    

How to run?    
--------
python DDPG_TORCS.py    (按F2切换到第一人称视角)     


The actions:
-------
输出action有三个维度：    
        - steer: 方向, 取值范围 [-1,1]    
        - accel： 油门，取值范围 [0,1]    
        - brake: 刹车，取值范围 [0,1]     


The reward has been define in the file gym_torcs_DDPG.py at line 129

The states from gym_torcs:    
s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) # the states   


Actor Network:
Let’s first talk about how to build the Actor Network in Keras. Here we used 2 hidden layers with 300 and 600 hidden units respectively. The output consist of 3 continuous actions, Steering, which is a single unit with tanh activation function (where -1 means max right turn and +1 means max left turn). Acceleration, which is a single unit with sigmoid activation function (where 0 means no gas, 1 means full gas). Brake, another single unit with sigmoid activation function (where 0 means no brake, 1 bull brake)   

Critic Network:   
The construction of the Critic Network is very similar to the Deep-Q Network in the previous post. The only difference is that we used 2 hidden layers with 300 and 600 hidden units. Also, the critic network takes both the states and the action as inputs. According to the DDPG paper, the actions were not included until the 2nd hidden layer of Q-network. Here we used the Keras function Merge to merge the action and the hidden layer together    

Target Network:
It is a well-known fact that directly implementing Q-learning with neural networks proved to be unstable in many environments including TORCS. Deepmind team came up the solution to the problem is to use a target network, where we created a copy of the actor and critic networks respectively, that are used for calculating the target values. The weights of these target networks are then updated by having them slowly track the learned networks:
θ′←τθ+(1−τ)θ′
where τ≪1. This means that the target values are constrained to change slowly, greatly improving the stability of learning.

Design of the rewards:     
Rt=Vxcos(θ)−Vxsin(θ)−Vx∣trackPos∣
In plain English, we want to maximum longitudinal velocity (first term), minimize transverse velocity (second term), and we also penalize the AI if it constantly drives very off center of the track (third term)


Design of the exploration algorithm
Another issue is how to design a right exploration algorithm in the continuous domain. In the previous blog post, we used ϵ greedy policy where the agent to try a random action some percentage of the time. However, this approach does not work very well in TORCS because we have 3 actions [steering,acceleration,brake]. If I just randomly choose the action from uniform random distribution we will generate some boring combinations [eg: the value of the brake is greater than the value of acceleration and the car simply not moving). Therefore, we add the noise using Ornstein-Uhlenbeck process to do the exploration.

Ornstein-Uhlenbeck process:   
What is Ornstein-Uhlenbeck process? In simple English it is simply a stochastic process which has mean-reverting properties.
dxt=θ(μ−xt)dt+σdWt here, θ means the how “fast” the variable reverts towards to the mean. μ represents the equilibrium or mean value. σ is the degree of volatility of the process. Interestingly, Ornstein-Uhlenbeck process is a very common approach to model interest rate, FX and commodity prices stochastically. (And a very common interview questions in finance quant interview). The following table shows the suggested values that I used in the code.

Experience Replay:    
Similar to the FlappyBird case, we also used the Experience Replay to saved down all the episode (s,a,r,s′) in a replay memory. When training the network, random mini-batches from the replay memory are used instead of most the recent transition, which will greatly improve the stability. 

Training:    
The actual training of the neural network is very simple, only contains 6 lines of code:   
        loss += critic.model.train_on_batch([states,actions], y_t)    
        a_for_grad = actor.model.predict(states)   
        grads = critic.gradients(states, a_for_grad)   
        actor.train(states, grads)   
        actor.target_train()  
        critic.target_train()    


总结一下：  
actor-critic框架是一个在循环的episode和时间步骤条件下，通过环境、actor和critic三者交互，来迭代训练策略网络、Q网络的过程


Reference:     
------
https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html     
https://arxiv.org/abs/1304.1672     
https://www.smwenku.com/a/5b7feb9c2b717767c6b2838a/    
https://blog.csdn.net/kenneth_yu/article/details/78781901    
