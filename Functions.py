import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture

#sets up the data from the excel sheet
def compile():
    df = pd.read_excel("/home/mackenzie/PycharmProjects/StartleTask/ARB Startle FINAL PP1-260 Cleaned_02.10.2017.xlsx", parse_cols= 'A, H, U, AG, AH') #parse_cols= 'C,E,H'
    subjects = np.array(df['Subject_ID'])
    stLat = np.array(df['Startle_Latency']) #takes in the latency data and transforms it into an array
    stLatUpd = [] #creates a blank list for updating with all good data
    stLatMeans = []  #creates a blank list for putting in each subject's latency mean
    stLatDevs = [] #creates a blank list for putting in each subject's latency st dev
    stAmp = np.array(df['Startle_Amp']) #takes in the amplitude data and transforms it into an array
    stAmpUpd = [] #creates a blank list for updating with all good data
    stAmpMeans = [] #creates a blank list for putting in each subject's amplitude mean
    stAmpDevs = [] #creates a blank list for putting in each subject's amplitude st dev
    preFilter = np.array(df['PreStartleFilter'])
    postFilter = np.array(df['PostStartleFilter'])
    trialCount = 0
    for x in range(0, 6208): #filters through all of the tasks/rows
        trialCount += 1
        #checks to make sure data is good and if not moves onto next subject
        if preFilter[x] == 1 & postFilter[x] == 1 & ~np.isnan(stAmp[x]): #checks to see if the data is good
            stAmpUpd.append(stAmp[x])
        else:
            trialCount == 24
        if preFilter[x] == 1 & postFilter[x] == 1 & ~np.isnan(stLat[x]):
            stLatUpd.append(stLat[x])
        else:
            trialCount == 24
        if trialCount== 24: #if this is the last task for a subject then we need to make conclusions
            if not np.isnan(np.mean(stLatUpd)):
                stLatMeans.append(np.mean(stLatUpd))
            if not np.isnan(np.std(stLatUpd)):
                stLatDevs.append(np.std(stLatUpd))
            if not np.isnan(np.mean(stAmpUpd)):
                stAmpMeans.append(np.mean(stAmpUpd))
            if not np.isnan(np.std(stAmpUpd)):
                stAmpDevs.append(np.std(stAmpUpd))
            stLatUpd = [] #empties the list so the next subject will fill them up
            stAmpUpd = []
            trialCount = 0
    return stLatMeans, stLatDevs, stAmpMeans, stAmpDevs

def GMM(X, titleb, titlec):
    lowest_bic = np.infty
    highest_bic = -1000000000000
    bic = []
    n_components_range = range(1, 7)
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] > 0:
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
        elif bic[-1] < 0 and not n_components == 1:
            if bic[-1] > highest_bic:
                highest_bic = bic[-1]
                best_gmm = gmm



    bic = np.array(bic)  # plot the BIC
    xvals = np.array([1,2,3,4,5,6])
    spl = plt.subplot(2,1,1)
    plt.plot(xvals, bic)
    plt.title(titleb)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []
    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(X)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                               color_iter)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.title(titlec)
    plt.show()

#sum of the squared error
def SSE(centroids, array, labels):
    sse = 0
    for i in range(centroids.shape[0]):  # starts at 0
        for j in range(array.shape[1]):
            if labels[j] == i:
                sse = sse + (centroids[i, 0] - array[0, j]) ** 2 + (centroids[i, 1] - array[1, j]) ** 2
    return sse

#clustering kmeans
def kmeans(data, k,): #will I need to send the datalocation
    dataN = np.array(data)
    clf = KMeans(n_clusters=k)
    clf.fit(dataN.T)
    labels_pr_ = clf.predict(dataN.T)
    centroids = clf.cluster_centers_
    labels = clf.labels_
    return dataN, centroids, labels, labels_pr_

#plotsclusters
def plotClusters(title, data, labels, centroids):
    colors = ["g.", "r.", "c.", "y.", "m."]
    plt.figure()
    for i in range(data.shape[1]):
        plt.plot(data[0, i], data[1, i], colors[labels[i]], markersize=10)
    plt.title(title)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker="x", s=150, linewidths=10, zorder=10)
    plt.show()

#plots the SSE data
def plotSSE(dataX, dataY, title):
    plt.figure()
    plt.scatter(dataX, dataY, color= 'b', marker= 'o')
    plt.title(title)
    plt.show()

def findBestK(SSEpoints):
    max = 0
    for i in range(19):
        if abs(SSEpoints[i+1]-SSEpoints[i]) > max:
            max = abs(SSEpoints[i+1]-SSEpoints[i])
            kbest = i+2 #think through this logic about k
    return kbest