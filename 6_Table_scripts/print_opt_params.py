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
from scipy.stats import gmean
import glob

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

mydir = os.path.expanduser("~/GitHub/HAIs/")

df = pd.read_pickle(mydir + "1_data/WinsorizedZscores.pkl")
print(list(df))
print(df.head(), '\n\n')
#sys.exit()

#########################################################################################
########################## IMPORT HAI DATA ##############################################

hais = ['CAUTI', 'CLABSI', 'MRSA', 'CDIFF']

for hai in hais:
    
    pi_opt = []
    z_opt = []
    min_days = []
    max_days = []
    median_days = []
    
    iqr_days = []
    min_pd = []
    max_pd = []
    median_pd = []
    iqr_pd = []
    
    fdates = ['2014-07-17', '2014-10-23', '2014-12-18', '2015-01-22', '2015-04-16', '2015-05-06', '2015-07-16', '2015-10-08', '2015-12-10', '2016-05-04', '2016-08-10', '2016-11-10', '2017-10-24', '2018-01-26', '2018-05-23', '2018-07-25', '2018-10-31', '2019-03-21', '2019-04-24', '2019-07-02', '2019-10-30', '2020-01-29', '2020-04-22']
    #fdates = [fdates[-1]]
    
    tdf = []
    for d in fdates:
        
        tdf = df[(df['HAI'] == hai) & (df['file date'] == d)]
        #if tdf.shape[0] == 0:
        #    continue
        
        p = tdf['pi_opt'].iloc[0]
        pi_opt.append(p)
        
        z = tdf['z_opt'].iloc[0]
        z_opt.append(z)
        
        min_d = np.nanmin(tdf['Days'])
        max_d = np.nanmax(tdf['Days'])
        median_d = np.nanmedian(tdf['Days'])
        min_days.append(min_d)
        max_days.append(max_d)
        median_days.append(median_d)
        
        p_d = tdf['Days'] / (tdf['Days'] + tdf['z_opt'])
        p25 = np.percentile(p_d, 25)
        p75 = np.percentile(p_d, 75)
        iqr_pd.append([p25, p75])
        
        min_pd.append(np.nanmin(p_d))
        max_pd.append(np.nanmax(p_d))
        median_pd.append(np.nanmedian(p_d))
        
        
    print('\n')
    print(hai)
    print('pi_opt:', np.nanmean(pi_opt), np.nanstd(pi_opt))
    print('z_opt:', np.nanmean(z_opt), np.nanstd(z_opt))
    
    print('Days:')
    print('min days:', np.nanmean(min_days), np.nanstd(min_days))
    print('max days:', np.nanmean(max_days), np.nanstd(max_days))
    print('median days:', np.nanmean(median_days), np.nanstd(median_days))
    
    print('Probabilities of detection:')
    print('min pd:', np.nanmean(min_pd), np.nanstd(min_pd))
    print('max pd:', np.nanmean(max_pd), np.nanstd(max_pd))
    print('median pd:', np.nanmean(median_pd), np.nanstd(median_pd))
    
    if fdates == ['2020-04-22']:
        print('pd IQR (2020-04-22):', p25, p75)
    
print('\n')