import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from math import pi
import sys
import os
import scipy as sc
import warnings
from scipy.stats import binned_statistic
from numpy import log10, sqrt
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.patches as patches
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from scipy.stats.kde import gaussian_kde


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

mydir = os.path.expanduser("~/GitHub/HAIs/")

d = 'Beds, Total Facility'


#########################################################################################
################################ FUNCTIONS ##############################################


#########################################################################################
########################## IMPORT HAI DATA ##############################################

CAUTI_df = pd.read_pickle(mydir + "data/CAUTI_Data_opt_for_SIRs.pkl")
CAUTI_df = CAUTI_df[CAUTI_df['CAUTI Predicted Cases'] >= 1]
CAUTI_df = CAUTI_df[CAUTI_df['CAUTI Urinary Catheter Days'] > 0]
CAUTI_df = CAUTI_df[CAUTI_df['simulated O/E'] >= 0]
CAUTI_df = CAUTI_df[CAUTI_df['O/E'] >= 0]

print('CAUTI: ', CAUTI_df.shape[0], 'rows')
#print(CAUTI_df.head())

CLABSI_df = pd.read_pickle(mydir + "data/CLABSI_Data_opt_for_SIRs.pkl")
CLABSI_df = CLABSI_df[CLABSI_df['CLABSI Predicted Cases'] >= 1]
CLABSI_df = CLABSI_df[CLABSI_df['CLABSI Number of Device Days'] > 0]
CLABSI_df = CLABSI_df[CLABSI_df['simulated O/E'] >= 0]
CLABSI_df = CLABSI_df[CLABSI_df['O/E'] >= 0]

print('CLABSI: ', CLABSI_df.shape[0], 'rows')
#print(CAUTI_df.head())

MRSA_df = pd.read_pickle(mydir + "data/MRSA_Data_opt_for_SIRs.pkl")
MRSA_df = MRSA_df[MRSA_df['MRSA Predicted Cases'] >= 1]
MRSA_df = MRSA_df[MRSA_df['MRSA patient days'] > 0]
MRSA_df = MRSA_df[MRSA_df['simulated O/E'] >= 0]
MRSA_df = MRSA_df[MRSA_df['O/E'] >= 0]

print('MRSA: ', MRSA_df.shape[0], 'rows')
#print(CAUTI_df.head())

CDIFF_df = pd.read_pickle(mydir + "data/CDIFF_Data_opt_for_SIRs.pkl")
CDIFF_df = CDIFF_df[CDIFF_df['CDIFF Predicted Cases'] >= 1]
CDIFF_df = CDIFF_df[CDIFF_df['CDIFF patient days'] > 0]
CDIFF_df = CDIFF_df[CDIFF_df['simulated O/E'] >= 0]
CDIFF_df = CDIFF_df[CDIFF_df['O/E'] >= 0]

print('CDIFF: ', CDIFF_df.shape[0], 'rows')
#print(CAUTI_df.head())

#########################################################################################
##################### DECLARE FIGURE 3A OBJECT ##########################################
#########################################################################################

#########################################################################################
################################ GENERATE FIGURE ########################################
#########################################################################################

################################## SUBPLOT 1 ############################################


################################## SUBPLOT 3 ############################################


################################## SUBPLOT 4 ############################################


################################## SUBPLOT 5 ############################################


################################## SUBPLOT 6 ############################################
