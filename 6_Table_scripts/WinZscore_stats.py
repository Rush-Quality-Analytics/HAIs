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

df = pd.read_pickle(mydir + "1_data/WinsorizedZscores.pkl")

print(list(df), '\n\n')

################################## CDIFF ############################################

def get_stats(df1, metric, dates):
    use_latest = False
    
    if dates == 'Most recent quarter':
        fdates = ['2020-04-22']

    else:
        fdates = ['2014-07-17', '2014-10-23', '2014-12-18', '2015-01-22', '2015-04-16', '2015-05-06', '2015-07-16', '2015-10-08', '2015-12-10', '2016-05-04', '2016-08-10', '2016-11-10', '2017-10-24', '2018-01-26', '2018-05-23', '2018-07-25', '2018-10-31', '2019-03-21', '2019-04-24', '2019-07-02', '2019-10-30', '2020-01-29', '2020-04-22']
    
    agg_df = pd.DataFrame(columns=list(df1) + ['worst quartile ' + metric, 'worst quartile SIR'])
    
    for fdate in fdates:

        tdf = df1[df1['file date'] == fdate]
        p75_sis = np.percentile(tdf['Winzorized z ' + metric], 75)
        p75_sir = np.percentile(tdf['Winzorized z SIR'], 75)
        
        tdf['worst quartile ' + metric] = p75_sis - tdf['Winzorized z ' + metric]
        tdf['worst quartile ' + metric] = tdf['worst quartile ' + metric].apply(lambda x : 0 if x >= 0 else 1)
        
        tdf['worst quartile SIR'] = p75_sir - tdf['Winzorized z SIR']
        tdf['worst quartile SIR'] = tdf['worst quartile SIR'].apply(lambda x : 0 if x >= 0 else 1)
        
        agg_df = pd.concat([agg_df, tdf], ignore_index=True)
    
    return agg_df
    
#########################################################################################
################################ TABLE STATS ############################################
#########################################################################################

file_dates = ['All quarters', 'Most recent quarter']
metrics = ['SIS', 'SIS']
HAI_ls = ['CAUTI', 'CLABSI', 'MRSA', 'CDIFF']

with open(mydir + "6_Table_scripts/output/WinZscore_Stats.txt", 'w+') as out:

    for dates in file_dates:
        out.write('--------------------------------------------------------------------------\n')
        out.write('--------------------------------  ' + dates + '\n')
        out.write('--------------------------------------------------------------------------\n\n')
        
        for metric in metrics:
            out.write('---------------------------  ' + metric + '  ---------------------------\n')
            for hai in HAI_ls:
                tdf = df[df['HAI'] == hai]
                tdf = get_stats(tdf, metric, dates)
                
                out.write('\n')
                out.write(hai + '\n\n')
                
                N = 100 * tdf[tdf[metric] <= 0].shape[0] / tdf.shape[0]
                out.write('Number of hospitals that beat their random expectation: ' + str(tdf[tdf[metric] <= 0].shape[0]) + '\n')
                out.write('Percent of hospitals that beat their random expectation: ' + str(N) + '\n')
                
                out.write('---------' + metric + '---------\n')
                ttdf = tdf[tdf['Winzorized z ' + metric] == np.nanmin(tdf['Winzorized z ' + metric])]
                out.write('Number of hospitals with min ' + metric +': ' + str(ttdf.shape[0]) + '\n')
                
                avg = []
                for i in range(100):
                    ttdf2 = ttdf.sample(n=1000, replace=True, random_state=0)
                    avg.append(np.mean(ttdf2['Days'].tolist()))
                out.write('Avg no. of days for hospitals with min ' + metric + ': ' + str(np.round(np.mean(avg),2)) + '\n')
                
                ttdf = tdf[(tdf[metric] <= 0) & (tdf['worst quartile ' + metric] == 1)]
                out.write('No. hospitals in the worst quartile that beat their random expectation: ' + str(ttdf.shape[0]) + '\n')
                ttdf = tdf[(tdf[metric] > 0) & (tdf['worst quartile ' + metric] == 0)]
                out.write('No. hospitals that failed beat their random expectation and were not in the worst quartile: ' + str(ttdf.shape[0]) + '\n')

                out.write('---------SIR---------\n')
                ttdf = tdf[tdf['Winzorized z SIR'] == np.nanmin(tdf['Winzorized z SIR'])]
                out.write('Number of hospitals with min SIR: ' + str(ttdf.shape[0]) + '\n')
                
                avg = []
                for i in range(100):
                    ttdf2 = ttdf.sample(n=1000, replace=True, random_state=0)
                    avg.append(np.mean(ttdf2['Days'].tolist()))
                out.write('Avg no. of days for hospitals with min SIS: ' + str(np.round(np.mean(avg), 2)) + '\n')
                
                ttdf = tdf[(tdf[metric] <= 0) & (tdf['worst quartile SIR'] == 1)]
                out.write('Number of hospitals in the worst quartile that beat their random expectation: ' + str(ttdf.shape[0]) + '\n')
                ttdf = tdf[(tdf[metric] > 0) & (tdf['worst quartile SIR'] == 0)]
                out.write('No. hospitals that failed beat their random expectation and were not in the worst quartile: ' + str(ttdf.shape[0]) + '\n')

                out.write('---------Quartile Change---------\n')
                ttdf = tdf[(tdf['worst quartile SIR'] == 1) & (tdf['worst quartile ' + metric] == 0)]
                out.write('Hospitals in worst SIR quartile but not in worst ' + metric + ' quartile: ' + str(ttdf.shape[0]) + '\n')
                ttdf = tdf[(tdf['worst quartile SIR'] == 0) & (tdf['worst quartile ' + metric] == 1)]
                out.write('Hospitals in worst ' + metric + ' quartile but not in worst SIR quartile: ' + str(ttdf.shape[0]) + '\n\n')
                
            out.write('\n\n\n')
            
    out.close()
