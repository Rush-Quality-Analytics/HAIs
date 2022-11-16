import pandas as pd
import numpy as np
import random
import sys
import os
import scipy as sc
import warnings
from scipy import stats
from scipy.stats import gmean
from scipy.stats.mstats import winsorize
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
from numpy import log10, sqrt


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

mydir = os.path.expanduser("~/GitHub/HAIs/")

############################################################################################
##################### Changes in Penalty Assignment ########################################
############################################################################################

main_df = pd.read_pickle(mydir + "1_data/WinsorizedZscores.pkl")
main_df = main_df[main_df['Predicted Cases'] >= 1]

#fdates = ['2014-07-17', '2014-10-23', '2014-12-18', '2015-01-22', '2015-04-16', '2015-05-06', '2015-07-16', '2015-10-08', '2015-12-10', '2016-05-04', '2016-08-10', '2016-11-10', '2017-10-24', '2018-01-26', '2018-05-23', '2018-07-25', '2018-10-31', '2019-03-21', '2019-04-24', '2019-07-02', '2019-10-30', '2020-01-29', '2020-04-22']
fdates = ['2020-01-29']

hais = ['CAUTI', 'CLABSI', 'MRSA', 'CDIFF']
hai_labels = ['CAUTI device days', 'CLABSI device days', 'MRSA patient days', 'CDIFF patient days']

for hai_i, hai in enumerate(hais):
    hi_ls = []
    lo_ls = []
    print('\n')
    print(hai)
    for fdate in fdates:
        
        tdf = main_df[(main_df['HAI'] == hai) & (main_df['file date'] == fdate)]
        tdf = tdf[tdf['Predicted Cases'] >= 1]
        
        # No. of hospitals
        print('No. of hospitals:', tdf.shape[0])
        
        # SIR max
        print('SIR max:', np.max(tdf['SIR']))
              
        # No. of hospitals with SIR of 0
        print('No. of hospitals with SIR of 0:', tdf[tdf['SIR'] == 0].shape[0])
        
        # mean SIR
        print('mean SIR:', np.nanmean(tdf['SIR']))
        # SD SIR
        print('SD:', np.nanstd(tdf['SIR']))
print('\n')
