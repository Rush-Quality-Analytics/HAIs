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


pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
warnings.filterwarnings('ignore')

mydir = os.path.expanduser("~/GitHub/HAIs/")


def get_btr(O, E, P):
    
    btr = []
    
    for i, exp_val in enumerate(E):
        obs_val = O[i]
        pred_val = P[i]
        if obs_val < exp_val and obs_val < pred_val:
            btr.append(1)
        elif obs_val >= exp_val or obs_val >= pred_val:
            btr.append(0)
        
    return btr


#########################################################################################
########################## IMPORT HAI DATA ##############################################

df = pd.read_pickle(mydir + "1_data/WinsorizedZscores.pkl")
df = df[df['Predicted Cases'] >= 1]

hais = ['CAUTI', 'CLABSI', 'MRSA', 'CDIFF']
fdates = ['2014-07-17', '2014-10-23', '2014-12-18', '2015-01-22', '2015-04-16', '2015-05-06', '2015-07-16', '2015-10-08', '2015-12-10', '2016-05-04', '2016-08-10', '2016-11-10', '2017-10-24', '2018-01-26', '2018-05-23', '2018-07-25', '2018-10-31', '2019-03-21', '2019-04-24', '2019-07-02', '2019-10-30', '2020-01-29', '2020-04-22']
#fdates = ['2020-04-22']

for hai in hais:
    print('\n', hai)
    
    sir_ls = []
    sis_ls = []
    
    for fdate in fdates:

        tdf = df[(df['HAI'] == hai) & (df['file date'] == fdate)]
            
        #########################################################################################
        ######################## DECLARE FIGURE OBJECT ##########################################
        #########################################################################################
        
        fig = plt.figure(figsize=(10, 10))
        rows, cols = 2, 2
        fs = 14
        sz = 50
        a = 0.8
        ec1 = '0.4'
        ec2 = '0.6'
        lw = 0.3
        num_bins = 60    
        
        ################################## SUBPLOT 1 ############################################
            
        scores1 = []
        scores2 = []
        scores = tdf['SIR'].tolist()
        min_x = min(scores)
        max_x = max(scores)
        
        O = tdf['Observed Cases'].tolist()
        E = tdf['expected O'].tolist()
        P = tdf['Predicted Cases'].tolist()
        
        btr = get_btr(O, E, P)
        for i, x in enumerate(btr):
            if x == 1:
                scores1.append(scores[i])
            else:
                scores2.append(scores[i])
        
        num_beat = []
        for s2 in scores2:
            n = 0
            for s1 in scores1:
                if s2 < s1:
                    n += 1
                    break
            if n > 0:
                num_beat.append(n)
        
        txt = str()
        if len(num_beat) > 0:  
            txt = str(len(num_beat)) + " hospitals with more HAIs than \npredicted or expected at random \nhad lower SIRs than higher \nperforming hospitals."
        else:
            txt = "No hospitals with more HAIs \nthan expected at random or predicted \nhad lower SIRs than \nhigher performing hospitals."
        
        sir_ls.append(len(num_beat))
        
        if fdate == '2020-04-22':
            ax1 = plt.subplot2grid((rows, cols), (0, 0), colspan=1, rowspan=1)
            counts2, bins2, bars2 = plt.hist(scores2, bins=np.linspace(min_x, max_x, num_bins), histtype='stepfilled', density=False, color='0.6', label='observed ≥ random or predicted', linewidth=2)
            counts1, bins1, bars1 = plt.hist(scores1, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='k', label='observed < random & predicted', linewidth=2)
            plt.text(0.07*np.max(scores), 0.8*np.max(counts1), txt, fontsize=fs-3)
            plt.ylabel('No. of hospitals', fontsize=fs+2)
            plt.xlabel('SIR, ' + hai, fontsize=fs+2)
            plt.tick_params(axis='both', labelsize=fs-2)
            plt.legend(bbox_to_anchor=(-0.04, 1.02, 2.48, .2), loc=10, ncol=2, mode="expand",prop={'size':fs-1})
        
        
        ################################## SUBPLOT 2 ############################################
            
        scores1 = []
        scores2 = []
        scores = tdf['SIS'].tolist()
        min_x = min(scores)
        max_x = max(scores)

        
        O = tdf['Observed Cases'].tolist()
        E = tdf['expected O'].tolist()
        P = tdf['Predicted Cases'].tolist()
        
        btr = get_btr(O, E, P)
        for i, x in enumerate(btr):
            if x == 1:
                scores1.append(scores[i])
            else:
                scores2.append(scores[i])
        
        num_beat = []
        for s2 in scores2:
            n = 0
            for s1 in scores1:
                if s2 < s1:
                    n += 1
            if n > 0:
                num_beat.append(n)
        
        txt = str()
        if len(num_beat) > 0:  
            txt = str(len(num_beat)) + " hospitals with more \nHAIs than predicted or \nexpected at random had \nlower SISs than higher \nperforming hospitals."
        else:
            txt = str(0) + " hospitals with more \nHAIs than predicted or \nexpected at random had \nlower SISs than higher \nperforming hospitals."
            
        sis_ls.append(len(num_beat))
        
        if fdate == '2020-04-22':
            ax2 = plt.subplot2grid((rows, cols), (0, 1), colspan=1, rowspan=1)
            counts2, bins2, bars2 = plt.hist(scores2, bins=np.linspace(min_x, max_x, num_bins), histtype='stepfilled', density=False, color='0.6', label='observed >= random or predicted', linewidth=2)
            counts1, bins1, bars1 = plt.hist(scores1, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='k', label='observed < random & predicted', linewidth=2)
            plt.xlim(-2, 5.2)
            plt.text(0.15*np.max(scores), 0.75*np.max(counts1), txt, fontsize=fs-3)
            plt.ylabel('No. of hospitals', fontsize=fs+2)
            plt.xlabel('SIS, ' + hai, fontsize=fs+2)        
            plt.tick_params(axis='both', labelsize=fs-2)
            #########################################################################################
            ################################ FINAL FORMATTING #######################################
            #########################################################################################
                
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            plt.savefig(mydir+'/7_figures/Fig3/Fig3_Hists_' + hai + '_' + fdate + '.eps', dpi=400, format="eps", bbox_inches = "tight")
            plt.close()
        
            print('SIR:', np.nanmean(sir_ls), ',', np.nanstd(sir_ls))
            print('SIS:', np.nanmean(sis_ls), ',', np.nanstd(sis_ls))
            