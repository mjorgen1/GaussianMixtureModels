#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
#from Functions import *
from sklearn import mixture

#DATASET
numComponents = 5
numPoints = 50
numTotalEle = numComponents*numPoints
dim = 1 #dimensions/features
dia = np.array([5, 8, 10, 12, 15])
sideLength = 500  #range

#creates new arrays to store the data, means, and covs
data = np.empty([numComponents*numPoints, dim])
meanCollect = np.empty(shape = 5)
covCollect = np.empty(shape = 5)
meanTruth = np.empty(shape = 5)
covTruth = np.empty(shape=5)

#generates test data
for i in np.arange(numComponents):
    mean = np.random.uniform(0, 20, dim)
    meanCollect[i] = mean
    cov = np.identity(dim) * (dia[i])
    covCollect[i]= cov
    points = np.random.multivariate_normal(mean, cov, numPoints-1)
    data[i*numPoints+1:(i+1)*numPoints, :] = points

print "The means from this data set are " , meanCollect
print "The covariances from this data set are " , covCollect

# np.float32
X = np.array(data, dtype=np.float32) #shape is 250,1

#plotting the 1D data
colors = np.array(['blue', 'green', 'red', 'yellow', 'purple'])
colorCount = 0
plt.figure()
for i in range(250):
    plt.scatter(X[i], 0, color = colors[colorCount])
    if i == 49 or i == 99 or i== 149 or i==  199:
        colorCount += 1
plt.show()


# Fit a Gaussian mixture with EM using five components
#gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)
#plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0, 'Gaussian Mixture')
# Fit a Dirichlet process Gaussian mixture using five components
#dpgmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type='full').fit(X)
#plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1, 'Bayesian Gaussian Mixture with a Dirichlet process prior')


#plots an animated line graph
# fig1 = plt.figure()
#
# data = np.random.rand(2, 25)
# l, = plt.plot([], [], 'r-')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel('x')
# plt.title('test')
# line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data,l),interval=50, blit=True)
# plt.show()

#then try 2d data
#try synthetic data




