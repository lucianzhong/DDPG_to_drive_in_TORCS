Requirements:
We are assuming you are using Ubuntu 14.04 LTS/16.04 LTS machine and installed
Python 3
Keras 1.1.0
Tensorflow r0.10
gym_torcs


How to install?
https://blog.csdn.net/ss910/article/details/77618425
(need to use sudo when run python snakeoil3_gym.py)

How to run?
python DDPG_TORCS.py    (按F2切换到第一人称视角)



The actions:
输出action有三个维度： 
- steer: 方向, 取值范围 [-1,1] 
- accel： 油门，取值范围 [0,1] 
- brake: 刹车，取值范围 [0,1]

The states from gym_torcs:
s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) # the states




总结一下： 
actor-critic框架是一个在循环的episode和时间步骤条件下，通过环境、actor和critic三者交互，来迭代训练策略网络、Q网络的过程











Reference:

https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html

https://www.smwenku.com/a/5b7feb9c2b717767c6b2838a/

https://blog.csdn.net/kenneth_yu/article/details/78781901
