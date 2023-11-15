# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 04:10:13 2022

@author: kohei
"""

#CSV読み込み
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix 
    
data1 = pd.read_csv("sampleData.csv")
#data1.head()
#print(data1)
data1.describe()

plt.scatter(data1['x'], data1['y'])
plt.ylabel('x')
plt.xlabel('y')
plt.title("correlation") #タイトル
plt.xlabel("Average Temperature of SAITAMA") #x軸のラベル
plt.ylabel("Average Temperature of IWATE") #y軸のラベル
plt.grid() #グリッド線を引く(引かなくてもいい別に)
plt.show()

#分位点回帰
import numpy as np
np.random.seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm

def QuantileGradientDescent(X, y, init_theta, tau, lr=1e-4, num_iters=10000):
    theta = init_theta
    for i in range(num_iters):
        y_hat = X @ theta # predictions
        delta = y - y_hat  # error
        indic = np.array(delta <= 0., dtype=np.float32) # indicator
        grad = np.abs(tau - indic) * np.sign(delta) # gradient
        theta += lr * X.T @ grad # Update
    return theta

def gaussian_func(x, mu, sigma):
    return (0.8/sigma)*np.exp( - (x - mu)**2 / (2 * sigma**2))

cmap = plt.cm.viridis(np.linspace(0., 1., 3))

# Generate Toy datas
N = 100 # sample size
x = data1['x']
y = data1['y']

#print(x)

X = np.ones((N, 2)) # design matrix
X[:,1] = x

taus = np.array([0.1, 0.5, 0.95])
m = len(taus) 
Y = np.zeros((m, N)) # memory array


for i in tqdm(range(m)):
    init_theta = np.zeros(2) # init variables
    theta = QuantileGradientDescent(X, y, init_theta, tau=taus[i])
    y_hat = X @ theta
    Y[i] = y_hat
    
# Results plot
plt.figure(figsize=(5,5))
plt.title("Quantile Regression")
plt.scatter(x, y, color="gray", s=5) # samples

for i in range(m):
    plt.plot([min(x), max(x)], [min(Y[i]), max(Y[i])], linewidth=2,
              color=cmap[i], label=str(int(taus[i]*100))+"%tile")  # regression line

for loc in range(1,5):
    noise_y = np.arange(0, 6*loc, 1e-3)
    noise_x = loc + gaussian_func(noise_y, 3*loc, loc)
    plt.fill_between(noise_x, -1, noise_y, color='#539ecd',
                     linewidth=2, alpha=0.5)
    plt.plot(noise_x, noise_y, color='#539ecd', linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(0, 25)
plt.legend()
plt.tight_layout()
plt.show()



