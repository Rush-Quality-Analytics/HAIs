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

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

mydir = os.path.expanduser("~/GitHub/HAIs/")

#########################################################################################
########################## IMPORT HAI DATA ##############################################

df = pd.read_pickle(mydir + "data/WinsorizedZscores.pkl")

print(list(df), '\n\n')

fdates = ['2014-07-17', '2014-10-23', '2014-12-18', '2015-01-22', '2015-04-16', '2015-05-06', '2015-07-16', '2015-10-08', '2015-12-10', '2016-05-04', '2016-08-10', '2016-11-10', '2017-10-24', '2018-01-26', '2018-05-23', '2018-07-25', '2018-10-31', '2019-03-21', '2019-04-24', '2019-07-02', '2019-10-30', '2020-01-29', '2020-04-22']

def agg_df(df1, fdates):
    agg_df = pd.DataFrame(columns=list(df1) + ['worst quartile SISc', 'worst quartile SIR'])
    
    for date in fdates:

        tdf = df1[df1['file date'] == date]
        p75_sis = np.percentile(tdf['Winzorized z SISc'], 75)
        p75_sir = np.percentile(tdf['Winzorized z SIR'], 75)
        
        tdf['worst quartile SISc'] = p75_sis - tdf['Winzorized z SISc']
        tdf['worst quartile SISc'] = tdf['worst quartile SISc'].apply(lambda x : 0 if x >= 0 else 1)
        
        tdf['worst quartile SIR'] = p75_sir - tdf['Winzorized z SIR']
        tdf['worst quartile SIR'] = tdf['worst quartile SIR'].apply(lambda x : 0 if x >= 0 else 1)
        
        agg_df = pd.concat([agg_df, tdf], ignore_index=True)
        
    return agg_df
    
    
#########################################################################################
################################ TABLE STATS ############################################
#########################################################################################

HAI_ls = ['CAUTI', 'CLABSI', 'MRSA', 'CDIFF']

with open(mydir + "data/WinZscore_Stats_Avg.txt", 'w+') as out:

    for hai in HAI_ls:
        out.write('--------------' + hai + '--------------')
        out.write('\n')
        
        tdf = df[df['HAI'] == hai]
        tdf = agg_df(tdf, fdates)
        #print(list(tdf))
        #sys.exit()
        
        n1 = [] # Number of hospitals that beat their random expectation
        
        n2 = [] # Percent of hospitals that beat their random expectation
        
        n3 = [] # Number of hospitals with min WinZ SIR score
        
        n4 = [] # Number of hospitals with min WinZ SIS score
        
        n5 = [] # Avg no. of days for hospitals with min SIR
        
        n6 = [] # Avg no. of days for hospitals with min SIS
        
        n7 = [] # SIR: No. hospitals in the worst quartile that beat their random expectation
        n7b = [] # SIR: % hospitals that beat their random expectation but were in the worst quartile
        
        n8 = [] # SIR: No. hospitals that failed beat their random expectation and were not in the worst quartile
        n8b = [] # SIR: % hospitals that faild to beat their random expectation but were not in the worst quartile
        
        n9 = [] # SIS: No. hospitals in the worst quartile that beat their random expectation
        
        n10 = [] #SIS: No. hospitals that failed beat their random expectation and were not in the worst quartile
        
        n11 = [] # No. Hospitals in worst SIR quartile but not in worst SIS quartile
        
        n12 = [] # No. Hospitals in worst SIS quartile but not in worst SIR quartile
        
        n13 = [] # % of hospitals that beat their random expectation that were in the worst quartile
        
        n14 = [] # number of hospitals with avg days or greater having SIR = 0
        
        n15 = [] # % of hospitals with avg days or greater that had an SIR = 0
        
        n16 = [] # % of hospitals with less than avg days that had an SIR = 0
        
        for date in fdates:
            tdf2 = tdf[tdf['file date'] == date]
            
            n = tdf2[tdf2['SISc'] < 0].shape[0]
            n1.append(n)
            
            n = 100 * tdf2[tdf2['SISc'] < 0].shape[0] / tdf2.shape[0]
            n2.append(n)
            
            sirs = tdf2['Winzorized z SIR'].tolist()
            min_sir = np.min(sirs)
            n = sirs.count(min_sir)
            n3.append(n)
            
            siss = tdf2['Winzorized z SISc'].tolist()
            min_sis = np.min(siss)
            n = siss.count(min_sis)
            n4.append(n)
            
            n = np.nanmedian(tdf2[tdf2['Winzorized z SIR'] == min_sir]['Days'])
            n5.append(n)
            
            n = np.nanmedian(tdf2[tdf2['Winzorized z SISc'] == min_sis]['Days'])
            n6.append(n)
            
            tdf3 = tdf2[(tdf2['SISc'] < 0) & (tdf2['worst quartile SIR'] == 1)]
            n7.append(tdf3.shape[0])
            n7b.append(100 * tdf3.shape[0]/tdf2[tdf2['SISc'] < 0].shape[0])
            
            tdf3 = tdf2[(tdf2['SISc'] >= 0) & (tdf2['worst quartile SIR'] == 0)]
            n8.append(tdf3.shape[0])
            n8b.append(100 * tdf3.shape[0]/tdf2[tdf2['SISc'] >= 0].shape[0])
            
            tdf3 = tdf2[(tdf2['SISc'] < 0) & (tdf2['worst quartile SISc'] == 1)]
            n9.append(tdf3.shape[0])
            
            tdf3 = tdf2[(tdf2['SISc'] >= 0) & (tdf2['worst quartile SISc'] == 0)]
            n10.append(tdf3.shape[0])
            
            tdf3 = tdf2[(tdf2['worst quartile SIR'] == 1) & (tdf2['worst quartile SISc'] == 0)]
            n11.append(tdf3.shape[0])
            
            tdf3 = tdf2[(tdf2['worst quartile SIR'] == 0) & (tdf2['worst quartile SISc'] == 1)]
            n12.append(tdf3.shape[0])
            
            tdf3 = tdf2[(tdf2['SISc'] < 0) & (tdf2['worst quartile SIR'] == 1)]
            n13.append(100 * tdf3.shape[0]/tdf2[(tdf2['SISc'] < 0)].shape[0])
            
            avg_days = np.nanmedian(tdf2['Days'])
            tdf3 = tdf2[tdf2['Days'] >= avg_days]
            SIRs = tdf3['O/E'].tolist()
            ct = SIRs.count(0)
            n14.append(ct)
            n15.append(100 * ct/len(SIRs))
            
            tdf3 = tdf2[tdf2['Days'] < avg_days]
            SIRs = tdf3['O/E'].tolist()
            ct = SIRs.count(0)
            n16.append(100 * ct/len(SIRs))
            
            
        avg = str(np.round(np.nanmean(n1),3))
        std = str(np.round(np.nanstd(n1),3))
        out.write('Avg no. of hospitals that beat their random expectation: ' + avg + ' ± ' + std)
        out.write('\n')
        
        avg = str(np.round(np.nanmean(n2),3))
        std = str(np.round(np.nanstd(n2),3))
        out.write('Avg percent of hospitals that beat their random expectation: ' + avg + ' ± ' + std)
        out.write('\n')
        out.write('\n')
        
        avg = str(np.round(np.nanmean(n3),3))
        std = str(np.round(np.nanstd(n3),3))
        out.write('Avg no. of hospitals with min WinZ SIR score: ' + avg + ' ± ' + std)
        out.write('\n')
        
        avg = str(np.round(np.nanmean(n4),3))
        std = str(np.round(np.nanstd(n4),3))
        out.write('Avg no. of hospitals with min WinZ SIS score: ' + avg + ' ± ' + std)
        out.write('\n')
        out.write('\n')
        
        avg = str(np.round(np.nanmean(n5),3))
        std = str(np.round(np.nanstd(n5),3))
        out.write('Avg median no. of days for hospitals with min SIR: ' + avg + ' ± ' + std)
        out.write('\n')
        
        avg = str(np.round(np.nanmean(n6),3))
        std = str(np.round(np.nanstd(n6),3))
        out.write('Avg median no. of days for hospitals with min SIS: ' + avg + ' ± ' + std)
        out.write('\n')
        out.write('\n')
        
        total = str(np.sum(n7))
        out.write('SIR: No. of times that hospitals in the worst quartile beat their random expectation: ' + total)
        out.write('\n')
        avg = str(np.round(np.nanmean(n7),3))
        std = str(np.round(np.nanstd(n7),3))
        out.write('SIR: Avg no. of hospitals in the worst quartile that beat their random expectation: ' + avg + ' ± ' + std)
        out.write('\n')
        avg = str(np.round(np.nanmean(n7b),3))
        std = str(np.round(np.nanstd(n7b),3))
        out.write('SIR: Avg % of hospitals that beat their random expectation but were in the worst quartile: ' + avg + ' ± ' + std)
        out.write('\n')
        out.write('\n')
        
        total = str(np.sum(n8))
        out.write('SIR: No. of times hospitals failed to beat their random expectation and avoided the worst quartile: ' + total)
        out.write('\n')
        avg = str(np.round(np.nanmean(n8),3))
        std = str(np.round(np.nanstd(n8),3))
        out.write('SIR: Avg no. of hospitals that failed to beat their random expectation and were not in the worst quartile: ' + avg + ' ± ' + std)
        out.write('\n')
        avg = str(np.round(np.nanmean(n8b),3))
        std = str(np.round(np.nanstd(n8b),3))
        out.write('SIR: Avg % of hospitals that failed to beat their random expectation but were not in the worst quartile: ' + avg + ' ± ' + std)
        out.write('\n')
        out.write('\n')
        
        
        avg = str(np.round(np.nanmean(n9),3))
        std = str(np.round(np.nanstd(n9),3))
        out.write('SIS: Avg no. of hospitals in the worst quartile that beat their random expectation: ' + avg + ' ± ' + std)
        out.write('\n')
        
        avg = str(np.round(np.nanmean(n10),3))
        std = str(np.round(np.nanstd(n10),3))
        out.write('SIS: Avg no. of hospitals that failed to beat their random expectation and were not in the worst quartile: ' + avg + ' ± ' + std)
        out.write('\n')
        out.write('\n')
        
        avg = str(np.round(np.nanmean(n11),3))
        std = str(np.round(np.nanstd(n11),3))
        out.write('Avg no. of hospitals in worst SIR quartile but not in worst SIS quartile: ' + avg + ' ± ' + std)
        out.write('\n')
        
        avg = str(np.round(np.nanmean(n12),3))
        std = str(np.round(np.nanstd(n12),3))
        out.write('Avg no. of hospitals in worst SIS quartile but not in worst SIR quartile: ' + avg + ' ± ' + std)
        out.write('\n')
        out.write('\n')
        
        #avg = str(np.round(np.nanmean(n13),3))
        #std = str(np.round(np.nanstd(n13),3))
        #out.write('Avg % of hospitals that beat their random expectation but were in the worst SIR quartile: ' + avg + ' ± ' + std)
        #out.write('\n')
        
        total = str(np.sum(n14))
        out.write('No. of times that hospitals with avg days or greater had an SIR of 0: ' + total)
        out.write('\n')
        
        avg = str(np.round(np.nanmean(n14),3))
        std = str(np.round(np.nanstd(n14),3))
        out.write('Avg no. of hospitals having avg days or greater and an SIR of 0: ' + avg + ' ± ' + std)
        out.write('\n')
        out.write('\n')
        
        avg = str(np.round(np.nanmean(n15),3))
        std = str(np.round(np.nanstd(n15),3))
        out.write('Avg % of hospitals with median days or greater having an SIR of 0: ' + avg + ' ± ' + std)
        out.write('\n')
        min_ = str(np.round(np.nanmin(n15),3))
        max_ = str(np.round(np.nanmax(n15),3))
        out.write('Min - Max % of hospitals with median days or greater having an SIR of 0: ' + min_ + ' - ' + max_)
        out.write('\n')
        out.write('\n')
        
        avg = str(np.round(np.nanmean(n16),3))
        std = str(np.round(np.nanstd(n16),3))
        out.write('Avg % of hospitals with less than avg days having an SIR of 0: ' + avg + ' ± ' + std)
        out.write('\n')
        min_ = str(np.round(np.nanmin(n16),3))
        max_ = str(np.round(np.nanmax(n16),3))
        out.write('Min - Max % of hospitals with less than median days having an SIR of 0: ' + min_ + ' - ' + max_)
        out.write('\n')
        out.write('\n')
        out.write('\n')
        
    out.close()
