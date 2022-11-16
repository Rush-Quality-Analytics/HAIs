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

#print(list(main_df))
#sys.exit()

fdates = ['2014-07-17', '2014-10-23', '2014-12-18', '2015-01-22', '2015-04-16', '2015-05-06', '2015-07-16', '2015-10-08', '2015-12-10', '2016-05-04', '2016-08-10', '2016-11-10', '2017-10-24', '2018-01-26', '2018-05-23', '2018-07-25', '2018-10-31', '2019-03-21', '2019-04-24', '2019-07-02', '2019-10-30', '2020-01-29', '2020-04-22']
fdates = ['2020-04-22']

hais = ['CAUTI', 'CLABSI', 'MRSA', 'CDIFF']
hai_labels = ['CAUTI device days', 'CLABSI device days', 'MRSA patient days', 'CDIFF patient days']

for hai_i, hai in enumerate(hais):
    hi_ls = []
    lo_ls = []
    sir_0_ls = []
    
    for fdate in fdates:
        
        fig = plt.figure(figsize=(10, 10))
        rows, cols = 2, 2
        fs = 16
        radius = 200
        sz = 60
        #########################################################################################
        ################################ GENERATE FIGURE ########################################
        #########################################################################################
        
        ################################## SUBPLOT 1 ############################################
        tdf = main_df[(main_df['HAI'] == hai) & (main_df['file date'] == fdate)]
        tdf['Rank in SIR'] = tdf['SIR'].rank(axis=0,method='min',na_option='keep',ascending=True)
        tdf['Rank in SIS'] = tdf['SIS'].rank(axis=0, method='min',na_option='keep',ascending=True)
        tdf['change in rank'] = tdf['Rank in SIR'] - tdf['Rank in SIS']
        
        
        ax1 = plt.subplot2grid((rows, cols), (0, 0), colspan=1, rowspan=1)
        ax1.scatter([20000], [2000], facecolors = 'k', s = sz+20, edgecolors='0.6', linewidths=1,
                    label='SIR = 0')
        ax1.scatter([20000], [2000], facecolors = '0.9', s = sz+20, edgecolors='0.6', linewidths=1,
                    label='SIR > 0')
        
        tdf2 = tdf[tdf['SIR'] != 0]
        x = tdf2['Days']
        y2 = tdf2['Rank in SIR']
        ax1.scatter(x, y2, facecolors = '0.9', s = sz-10, edgecolors='0.6', linewidths=.7)
        
        tdf2 = tdf[tdf['SIR'] == 0]
        x = tdf2['Days']
        y2 = tdf2['Rank in SIR']
        ax1.scatter(x, y2, facecolors = 'k', s = sz-10, edgecolors='0.6', linewidths=.7)
        
        if hai == ['MRSA', 'CLABSI']:
            ax1.set_yticks([1, 500, 1000, 1500, 2000]) 
            ax1.set_yticklabels([1, 500, 1000, 1500, 2000])
        elif hai == 'CAUTI':
            ax1.set_yticks([1, 500, 1000, 1500, 2000, 2500]) 
            ax1.set_yticklabels([1, 500, 1000, 1500, 2000, 2500])
        elif hai == 'CDIFF':
            ax1.set_yticks([1, 500, 1000, 1500, 2000, 2500, 3000]) 
            ax1.set_yticklabels([1, 500, 1000, 1500, 2000, 2500, 3000])
            
        ax1.invert_yaxis()
        plt.ylabel('Rank in SIR', fontsize=fs+3)
        plt.xlabel(hai_labels[hai_i], fontsize=fs+2)
        plt.tick_params(axis='both', labelsize=fs-2)
        
        plt.legend(bbox_to_anchor=(-0.04, 1.02, 2.48, .2), loc=10, ncol=2, 
                   mode="expand",prop={'size':fs+4},
                   #handletextpad=0.0,
                   )
        
        plt.xscale('log')
        
        if hai == 'CDIFF':
            plt.text(400000, 3100, 'A', fontweight='bold', fontsize=fs+2)
        elif hai == 'CAUTI':
            plt.text(50000, 2300, 'A', fontweight='bold', fontsize=fs+2)
        elif hai == 'CLABSI':
            plt.text(1000, 200, 'A', fontweight='bold', fontsize=fs+2)
        elif hai == 'MRSA':
            plt.text(12000, 300, 'A', fontweight='bold', fontsize=fs+2)
        
        ################################## SUBPLOT 2 ############################################
        tdf = main_df[(main_df['HAI'] == hai) & (main_df['file date'] == fdate)]
        tdf['Rank in SIR'] = tdf['SIR'].rank(axis=0,method='min',na_option='keep',ascending=True)
        tdf['Rank in SIS'] = tdf['SIS'].rank(axis=0, method='min',na_option='keep',ascending=True)
        tdf['change in rank'] = tdf['Rank in SIR'] - tdf['Rank in SIS']
        
        ax2 = plt.subplot2grid((rows, cols), (0, 1), colspan=1, rowspan=1)
        
        tdf2 = tdf[tdf['SIR'] != 0]
        x = tdf2['Days']
        y2 = tdf2['Rank in SIS']
        ax2.scatter(x, y2, facecolors = '0.9', s = sz-10, edgecolors='0.6', linewidths=.7)
        
        tdf2 = tdf[tdf['SIR'] == 0]
        x = tdf2['Days']
        y2 = tdf2['Rank in SIS']
        ax2.scatter(x, y2, facecolors = 'k', s = sz-10, edgecolors='0.6', linewidths=.7)
        
        if hai == ['MRSA', 'CLABSI']:
            ax2.set_yticks([1, 500, 1000, 1500, 2000]) 
            ax2.set_yticklabels([1, 500, 1000, 1500, 2000])
        elif hai == 'CAUTI':
            ax2.set_yticks([1, 500, 1000, 1500, 2000, 2500]) 
            ax2.set_yticklabels([1, 500, 1000, 1500, 2000, 2500])
        elif hai == 'CDIFF':
            ax2.set_yticks([1, 500, 1000, 1500, 2000, 2500, 3000]) 
            ax2.set_yticklabels([1, 500, 1000, 1500, 2000, 2500, 3000])
            
        plt.ylabel('Rank in SIS', fontsize=fs+3)
        plt.xlabel(hai_labels[hai_i], fontsize=fs+2)
        plt.tick_params(axis='both', labelsize=fs-2)
        ax2.invert_yaxis()
        plt.xscale('log')
        
        if hai == 'CDIFF':
            plt.text(400000, 3100, 'B', fontweight='bold', fontsize=fs+2)
        elif hai == 'CAUTI':
            plt.text(50000, 2300, 'B', fontweight='bold', fontsize=fs+2)
        elif hai == 'CLABSI':
            plt.text(1000, 100, 'B', fontweight='bold', fontsize=fs+2)
        elif hai == 'MRSA':
            plt.text(12000, 150, 'B', fontweight='bold', fontsize=fs+2)
            
        ################################## SUBPLOT 3 ############################################
        tdf = main_df[(main_df['HAI'] == hai) & (main_df['file date'] == fdate)]
        tdf['Rank in SIR'] = tdf['SIR'].rank(axis=0,method='min',na_option='keep',ascending=True)
        tdf['Rank in SIS'] = tdf['SIS'].rank(axis=0, method='min',na_option='keep',ascending=True)
        tdf['change in rank'] = tdf['Rank in SIR'] - tdf['Rank in SIS']

        p10 = np.percentile(tdf['Days'], 10)
        tdf_lo = tdf[tdf['Days'] < p10]
        lo_ls.extend(tdf_lo['change in rank'])
        
        p90 = np.percentile(tdf['Days'], 90)
        tdf_hi = tdf[tdf['Days'] > p90]
        hi_ls.extend(tdf_hi['change in rank'])
        
        tdf2 = tdf[tdf['SIR'] == 0]
        sir_0_ls.extend(tdf2['change in rank'])
        
        ax2 = plt.subplot2grid((rows, cols), (1, 0), colspan=2, rowspan=1)
        
        tdf2 = tdf[tdf['SIR'] != 0]
        x = tdf2['Days']
        y2 = tdf2['change in rank']
        ax2.scatter(x, y2, facecolors = '0.9', s = sz-10, edgecolors='0.6', linewidths=.7)
        plt.hlines(0, 0, max(x), colors='k')
        xmax = max(x)
        
        tdf2 = tdf[tdf['SIR'] == 0]
        x = tdf2['Days']
        y2 = tdf2['change in rank']
        ax2.scatter(x, y2, facecolors = 'k', s = sz-10, edgecolors='0.6', linewidths=.7)
        
        if hai == 'CDIFF':
            plt.text(700, -1500, '       Change in Rank\n\nWorsened        Improved', 
                     fontsize=fs+1, rotation='90')
        elif hai == 'CAUTI':
            plt.text(130, -1100, '       Change in Rank\n\nWorsened        Improved', 
                     fontsize=fs+1, rotation='90')
        elif hai == 'CLABSI':
            plt.text(430, -600, '       Change in Rank\n\nWorsened        Improved', 
                     fontsize=fs+1, rotation='90')
        elif hai == 'MRSA':
            plt.text(6000, -670, '       Change in Rank\n\nWorsened        Improved', 
                     fontsize=fs+1, rotation='90')
            
        plt.xscale('log')
        plt.xlabel(hai_labels[hai_i], fontsize=fs+2)
        plt.tick_params(axis='both', labelsize=fs-2)
        #plt.title('Hospitals with the worst\nWinsorized SIR z-scores', fontsize = fs+4)
        plt.xlim(min(x), xmax)
        
        if hai == 'CDIFF':
            plt.text(2600, 2300, 'C', fontweight='bold', fontsize=fs+2)
        elif hai == 'CAUTI':
            plt.text(500, 1000, 'C', fontweight='bold', fontsize=fs+2)
        elif hai == 'CLABSI':
            plt.text(1200, 500, 'C', fontweight='bold', fontsize=fs+2)
        elif hai == 'MRSA':
            plt.text(15000, 930, 'C', fontweight='bold', fontsize=fs+2)
            
        if fdate == '2020-04-22':
            plt.subplots_adjust(wspace=0.4, hspace=0.25)
            plt.savefig(mydir+'/7_figures/Fig4/change_in_rank_' + hai + '_' + fdate + '.eps', format="eps", dpi=400, bbox_inches = "tight")
            plt.close()
    
    print(hai, 'SIRs of 0:', '|', np.nanmean(sir_0_ls), ',', np.nanstd(sir_0_ls))
    print(hai, 'low-volume:', '|', np.nanmean(lo_ls), ',', np.nanstd(lo_ls))
    print(hai, 'high-volume:', '|', np.nanmean(hi_ls), ',', np.nanstd(hi_ls), '\n')
     
        



