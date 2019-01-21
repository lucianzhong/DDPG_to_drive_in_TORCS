import random
import numpy as np 

# 使用Uhlenbeck-Ornstein随机过程（下面简称UO过程），作为引入的随机噪声 exploration

class OU(object):

    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)