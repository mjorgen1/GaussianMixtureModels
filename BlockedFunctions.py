import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import itertools
from scipy import linalg

#need to read through EDA_TotalSCRs (Col E), EDA_TonicSCL (Col H), HRV_MeanIBI (Col AU), IMP_LVET (Col BH),IMP_CO (Col BJ), IMP_PEP (Col BL)
def compile():
    #reads through the two excel sheets
    df = pd.read_excel("/home/mackenzie/data/ARB IAPSIADS Final Compiled All Physio Variables PP 1-260.xlsx", parse_cols= 'A, B, E, H, AU, BH, BJ, BL')
    df2 = pd.read_excel("/home/mackenzie/data/ARB Baselines Final Compiled All Physio Variables PP 1-260.xlsx", parse_cols= 'A, AE, C, E, H, U, BX, BZ, CB')

    #sets up the baseline excel sheet
    subjectsB = np.array(df2['EDA_SubjectID'])
    subjectTrack = []
    subjCountB = 1
    eventsB = np.array(df2['EDA_EventName'])
    eventCorrectB = 'IAPSIADSBL30'
    EDA_TotalSCRsB = np.array(df2['EDA_TotalSCRs'])
    EDA_TonicSCLB = np.array(df2['EDA_TonicSCL'])
    IMP_LVETB = np.array(df2['IMP_LVET'])
    IMP_COB = np.array(df2['IMP_CO'])
    IMP_PEPB = np.array(df2['IMP_PEP'])
    HRV_MeanIBIB = np.array(df2['HRV_MeanIBI'])
    Avg_EDA_TotalSCRsB = []
    Avg_EDA_TonicSCLB = []
    Avg_IMP_LVETB = []
    Avg_IMP_COB = []
    Avg_IMP_PEPB = []
    Avg_HRV_MeanIBIB = []
    done = False
    for i in range(0, 7253): #goes through every row of the baseline excel sheet
      if(eventsB[i]==eventCorrectB and subjectsB[i]==subjCountB and not done):
        if not np.isnan(EDA_TotalSCRsB[i]) and not np.isnan(EDA_TonicSCLB[i]) and not np.isnan(IMP_LVETB[i]) and not np.isnan(IMP_COB[i]) and not np.isnan(IMP_PEPB[i]) and not np.isnan(HRV_MeanIBIB[i]):
            Avg_EDA_TotalSCRsB.append(averageBaseline(EDA_TotalSCRsB, i))
            Avg_EDA_TonicSCLB.append(averageBaseline(EDA_TonicSCLB, i))
            Avg_IMP_LVETB.append(averageBaseline(IMP_LVETB, i))
            Avg_IMP_COB.append(averageBaseline(IMP_COB, i))
            Avg_IMP_PEPB.append(averageBaseline(IMP_PEPB, i))
            Avg_HRV_MeanIBIB.append(averageBaseline(HRV_MeanIBIB, i))
            done = True
            subjectTrack.append(subjectsB[i]) #subjectTrack and all the Avg lists should have the same # of elements
      if (i + 1 == 18129):
        break
      if(subjectsB[i+1] > subjCountB):
        subjCountB+=1
        done = False
    #assigns the lists to arrays
    subjectTrack_Arr = np.array(subjectTrack)
    Avg_EDA_TotalSCRsB_Arr = np.array(Avg_EDA_TotalSCRsB)
    Avg_EDA_TonicSCLB_Arr = np.array(Avg_EDA_TonicSCLB)
    Avg_IMP_LVETB_Arr = np.array(Avg_IMP_LVETB)
    Avg_IMP_COB_Arr = np.array(Avg_IMP_COB)
    Avg_IMP_PEPB_Arr = np.array(Avg_IMP_PEPB)
    Avg_HRV_MeanIBIB_Arr = np.array(Avg_HRV_MeanIBIB)
    subject_Tuple = subjectTrack_Arr.shape
    Total_Subjects = subject_Tuple[0]

    #prepares to filter through the experimental values for blocked pictures
    subjTrack_Count = 0
    Upd_Subject_Track = []
    subjects = np.array(df['Subject'])
    events = np.array(df['Event'])
    eventCorrect = 'IAPSBlStim'
    EDA_TotalSCRs = np.array(df['EDA_TotalSCRs'])
    EDA_TonicSCL = np.array(df['EDA_TonicSCL'])
    IMP_LVET = np.array(df['IMP_LVET'])
    IMP_CO = np.array(df['IMP_CO'])
    IMP_PEP = np.array(df['IMP_PEP'])
    HRV_MeanIBI = np.array(df['HRV_MeanIBI'])
    Final_EDA_TotalSCRs = []
    Final_EDA_TonicSCL = []
    Final_IMP_LVET = []
    Final_IMP_CO = []
    Final_IMP_PEP = []
    Final_HRV_MeanIBI = []

    #loops through every row of the baseline excel sheet and if a subject # and subjectTracker from the baseline align,
    # and the event is correct we calculate the data
    for i in range(0, 18101):
        if(subjects[i] == subjectTrack_Arr[subjTrack_Count] and events[i] == eventCorrect):
            if(not np.isnan(EDA_TotalSCRs[i]) and not np.isnan(EDA_TonicSCL[i]) and not np.isnan(IMP_LVET[i]) and not np.isnan(IMP_CO[i]) and not np.isnan(IMP_PEP[i]) and not np.isnan(HRV_MeanIBI[i])):
                Upd_Subject_Track.append(subjects[i])
                Final_EDA_TotalSCRs.append(averageValues(Avg_EDA_TotalSCRsB_Arr, subjTrack_Count, EDA_TotalSCRs, i))
                Final_EDA_TonicSCL.append(averageValues(Avg_EDA_TonicSCLB_Arr, subjTrack_Count, EDA_TonicSCL, i))
                Final_IMP_LVET.append(averageValues(Avg_IMP_LVETB_Arr, subjTrack_Count, IMP_LVET, i))
                Final_IMP_CO.append(averageValues(Avg_IMP_COB_Arr, subjTrack_Count, IMP_CO, i))
                Final_IMP_PEP.append(averageValues(Avg_IMP_PEPB_Arr, subjTrack_Count, IMP_PEP, i))
                Final_HRV_MeanIBI.append(averageValues(Avg_HRV_MeanIBIB_Arr, subjTrack_Count, HRV_MeanIBI, i))
            subjTrack_Count += 1
            if(subjTrack_Count+1 == Total_Subjects):
                break
    Upd_Subject_Track_Arr = np.array(Upd_Subject_Track)
    Final_EDA_TotalSCRs_Arr = np.array(Final_EDA_TotalSCRs)
    Final_EDA_TonicSCL_Arr = np.array(Final_EDA_TonicSCL)
    Final_IMP_LVET_Arr = np.array(Final_IMP_LVET)
    Final_IMP_CO_Arr = np.array(Final_IMP_CO)
    Final_IMP_PEP_Arr = np.array(Final_IMP_PEP)
    Final_HRV_MeanIBI_Arr = np.array(Final_HRV_MeanIBI)

    return Final_EDA_TotalSCRs_Arr, Final_EDA_TonicSCL_Arr, Final_IMP_LVET_Arr, Final_IMP_CO_Arr, Final_IMP_PEP_Arr, Final_HRV_MeanIBI_Arr, Upd_Subject_Track_Arr

#function helps compile function average baseline values
def averageBaseline(array, index):
    count = 0
    avCount = 0
    sum = 0
    while count < 6:
        if not np.isnan(array[index+count]):
            sum += array[index+count]
            avCount += 1
        count +=1
    average = sum/avCount
    return average

def averageValues(arrayBaselines, indexOfB, arrayValues, indexofVal):
    count = 0
    avCount = 0
    sum = 0
    while count < 5:
        if not np.isnan(arrayValues[indexofVal+count]):
            curr = arrayValues[indexofVal] - arrayBaselines[indexOfB]
            sum += curr
            avCount +=1
        count += 1
    average = sum/avCount
    return average

#ADD STANDARDIZATION OF DATA METHOD
def standardize_Data(data):
    mean = np.mean(data)
    stDev = np.std(data)
    norm_list = []
    size_tuple = data.shape
    size_int = size_tuple[0]
    for i in range(0, size_int-1):
        curr = (data[i]-mean)/stDev
        norm_list.append(curr)
    norm_arr = np.array(norm_list)
    return norm_arr

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
















