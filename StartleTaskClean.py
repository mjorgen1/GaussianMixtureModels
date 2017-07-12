from Functions import *
import numpy as np


#gets data from excel sheet
stLatMeans, stLatDevs, stAmpMeans, stAmpDevs = compile()

#sets the data for amplitude and latencies into two dim arrays
data_Amp = [stAmpMeans, stAmpDevs]
data_Lat = [stLatMeans, stLatDevs]

# Fit a Gaussian mixture with 242 samples, and 2 features
A = np.array(data_Amp).T
L = np.array(data_Lat).T
GMM(A, "Amplitude BIC", "Amplitude GMM")
GMM(L, "Latency BIC", "Latency GMM")







#array I will plot by K and SSE to find the best K for the data set
# K_SSE_Amp = np.zeros([20])
# K_SSE_Lat = np.zeros([20])
# K_vals = np.empty([20])
# for i in range(20): K_vals[i] = i+1
#
# #goes through the K_SSE vars and adds to the columns of them setting the sse to that, and printing out the row #s
# #runs kmeans for the data
# for k in range(20):
#     k+=1
#     data_Amp, centroids_Amp, labels_Amp, labels_pr_amp = kmeans(data_Amp, k)
#     data_Lat, centroids_Lat, labels_Lat, labels_pr_Lat = kmeans(data_Lat, k)
#     SSE_amp = SSE(centroids_Amp, data_Amp, labels_Amp)
#     SSE_lat = SSE(centroids_Lat, data_Lat, labels_Lat)
#     K_SSE_Amp[k-1] = SSE_amp
#     K_SSE_Lat[k-1] = SSE_lat
#     SSE_lat = SSE(centroids_Lat, data_Lat, labels_Lat)
#     print 'The k for Startle Amplitude', k, 'the SSE is ', SSE_amp
#     print 'The k for Startle Latency ', k , 'the SSE is ' , SSE_lat, '\n'
#
# #plots the sse and k data sets
# plotSSE(K_vals, K_SSE_Amp, 'Amplitude K values and SSE values')
# plotSSE(K_vals, K_SSE_Lat, 'Latency K values and SSE values')
#
# #tells user what the best k values are for the clustered data set
# print 'The best k for Startle Amplitude is ' , findBestK(K_SSE_Amp)
# print 'The best k for Startle Latency is ' , findBestK(K_SSE_Lat), '\n'
#
# #plots the data with the best possible k value
# data_AmpBest, centroids_AmpB, labels_AmpB, labels_pr_ampB = kmeans(data_Amp, findBestK(K_SSE_Amp))
# plotClusters('Amplitude Startle Task Data with Kmeans', data_AmpBest, labels_AmpB, centroids_AmpB)
# data_LatBest, centroids_LatB, labels_LatB, labels_pr_LatB = kmeans(data_Lat, findBestK(K_SSE_Lat))
# plotClusters('Latency Startle Task Data with Kmeans', data_LatBest, labels_LatB, centroids_LatB)