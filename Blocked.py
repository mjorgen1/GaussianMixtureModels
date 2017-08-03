#!/usr/bin/env python

from BlockedFunctions import*
import numpy as np

#GMM FOR BLOCKED PICTURE TASK

EDA_TotalSCRs, EDA_TonicSCL, IMP_LVET, IMP_CO, IMP_PEP, HRV_MeanIBI, Subjects_ID = compile()

Standardize_EDA_TotalSCRs = standardize_Data(EDA_TotalSCRs)
Standardize_EDA_TonicSCL = standardize_Data(EDA_TonicSCL)
Standardize_IMP_LVET = standardize_Data(IMP_LVET)
Standardize_IMP_CO = standardize_Data(IMP_CO)
Standardize_IMP_PEP = standardize_Data(IMP_PEP)
Standardize_HRV_MeanIBI = standardize_Data(HRV_MeanIBI)

a = Standardize_EDA_TotalSCRs.reshape(216, 1)
b = Standardize_EDA_TonicSCL.reshape(216, 1)
c = Standardize_IMP_LVET.reshape(216, 1)
d = Standardize_IMP_CO.reshape(216, 1)
e = Standardize_IMP_PEP.reshape(216, 1)
f = Standardize_HRV_MeanIBI.reshape(216, 1)
dataFinal = np.concatenate((a, b, c, d, e, f), axis=1)

GMM(dataFinal, 'BIC for Blocked Picture Task', 'GMM for Blocked Picture Task')