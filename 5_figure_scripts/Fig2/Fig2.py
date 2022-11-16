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
from numpy import log10, sqrt
from scipy import stats

warnings.filterwarnings('ignore')
mydir = os.path.expanduser("~/GitHub/HAIs/")

#########################################################################################
################################ FUNCTIONS ##############################################
#########################################################################################
def obs_pred_rsquare(obs, pred):
    # Determines the prop of variability in a data set accounted for by a model
    # In other words, this determines the proportion of variation explained by
    # the 1:1 line in an observed-predicted plot.
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)

def histogram_intersection(h1, h2):
    print(sum(h1), sum(h2))
    i1 = 100 * np.sum(np.minimum(np.array(h1)/sum(h1), np.array(h2)/sum(h2)))
    i2 = 100 * np.sum(np.minimum(np.array(h1), np.array(h2)))/sum(h2)
    return i1, i2
    

#########################################################################################
########################## IMPORT HAI DATA ##############################################
#########################################################################################

num_bins = 40

hi_CAUTI = []
hi_CLABSI = []
hi_CDIFF = []
hi_MRSA = []

fdates = ['2020-04-22']#'2014-07-17', '2014-10-23', '2014-12-18', '2015-01-22', '2015-04-16', '2015-05-06', '2015-07-16', '2015-10-08', '2015-12-10', '2016-05-04', '2016-08-10', '2016-11-10', '2017-10-24', '2018-01-26', '2018-05-23', '2018-07-25', '2018-10-31', '2019-03-21', '2019-04-24', '2019-07-02', '2019-10-30', '2020-01-29', '2020-04-22']

for fdate in sorted(fdates, reverse=False):
    print(fdate)
    
    CAUTI_df = pd.read_pickle(mydir + "1_data/optimized_by_quarter/CAUTI/CAUTI_Data_opt_for_SIRs_" + fdate + ".pkl")
    CAUTI_df = CAUTI_df[CAUTI_df['CAUTI Predicted Cases'] >= 1]
    CAUTI_df = CAUTI_df[CAUTI_df['CAUTI Urinary Catheter Days'] > 0]
    CAUTI_df = CAUTI_df[CAUTI_df['expected O/E'] >= 0]
    CAUTI_df = CAUTI_df[CAUTI_df['O/E'] >= 0]

    CLABSI_df = pd.read_pickle(mydir + "1_data/optimized_by_quarter/CLABSI/CLABSI_Data_opt_for_SIRs_" + fdate + ".pkl")
    CLABSI_df = CLABSI_df[CLABSI_df['CLABSI Predicted Cases'] >= 1]
    CLABSI_df = CLABSI_df[CLABSI_df['CLABSI Number of Device Days'] > 0]
    CLABSI_df = CLABSI_df[CLABSI_df['expected O/E'] >= 0]
    CLABSI_df = CLABSI_df[CLABSI_df['O/E'] >= 0]

    MRSA_df = pd.read_pickle(mydir + "1_data/optimized_by_quarter/MRSA/MRSA_Data_opt_for_SIRs_" + fdate + ".pkl")
    MRSA_df = MRSA_df[MRSA_df['MRSA Predicted Cases'] >= 1]
    MRSA_df = MRSA_df[MRSA_df['MRSA patient days'] > 0]
    MRSA_df = MRSA_df[MRSA_df['expected O/E'] >= 0]
    MRSA_df = MRSA_df[MRSA_df['O/E'] >= 0]

    CDIFF_df = pd.read_pickle(mydir + "1_data/optimized_by_quarter/CDIFF/CDIFF_Data_opt_for_SIRs_" + fdate + ".pkl")
    CDIFF_df = CDIFF_df[CDIFF_df['CDIFF Predicted Cases'] >= 1]
    CDIFF_df = CDIFF_df[CDIFF_df['CDIFF patient days'] > 0]
    CDIFF_df = CDIFF_df[CDIFF_df['expected O/E'] >= 0]
    CDIFF_df = CDIFF_df[CDIFF_df['O/E'] >= 0]

    #########################################################################################
    ##################### DECLARE FIGURE 3A OBJECT ##########################################
    #########################################################################################

    fig = plt.figure(figsize=(8, 8))
    rows, cols = 2, 2
    fs = 18
    radius = 2
    measure = 'simulated O/E'
    
    #########################################################################################
    ################################ GENERATE FIGURE ########################################
    #########################################################################################

    ################################## SUBPLOT 1 ############################################
    ax1 = plt.subplot2grid((rows, cols), (0, 0), colspan=1, rowspan=1)

    tdf = CAUTI_df.copy(deep=True)
    y = tdf['O/E']**0.5
    y = y.tolist()
    x = tdf[measure]**0.5
    x = x.tolist()
    x = sorted(x)
    y = sorted(y)

    min_x = min([min(x), min(y)])
    max_x = max([max(x), max(y)])

    counts2, bins2, bars2 = plt.hist(y, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='k', label='Actual SIRs', linewidth=2)
    counts1, bins1, bars1 = plt.hist(x, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='0.5', label='Random sampling', linewidth=1.5)
    hi, hi2 = histogram_intersection(counts1, counts2)
    print(hi, hi2)
    r2 = obs_pred_rsquare(counts2, counts1)
    print('CAUTI, p-val:', np.rint(hi), ' r2:', np.rint(r2))
    
    plt.text(0.45*max_x, 0.85*max([max(counts1), max(counts2)]), '∩ = ' + str(int(np.rint(hi))) + '%', fontsize=fs+3)
    plt.legend(loc=0, frameon=False, fontsize=fs)
    plt.legend(bbox_to_anchor=(-0.05, 1.05, 2.45, .2), loc=10, ncol=2, frameon=True, mode="expand",prop={'size':fs})
    
    plt.ylabel('No. of hospitals', fontsize=fs)
    plt.xlabel('SIR, CAUTI', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-3)
    ax1.set_xticks([0, 1, 2]) # choose which x locations to have ticks
    ax1.set_xticklabels([0, 1, 2**2]) # set the labels to display at those ticks
    
    hi_CAUTI.append(hi)
    
    ################################## SUBPLOT 2 ############################################

    ax2 = plt.subplot2grid((rows, cols), (0, 1), colspan=1, rowspan=1)
    tdf = CLABSI_df.copy(deep=True)
    y = tdf['O/E']**0.5
    y = y.tolist()
    x = tdf[measure]**0.5
    x = x.tolist()
    x = sorted(x)
    y = sorted(y)

    min_x = min([min(x), min(y)])
    max_x = max([max(x), max(y)])

    counts2, bins2, bars2 = plt.hist(y, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='k', label='Actual', linewidth=2)
    counts1, bins1, bars1 = plt.hist(x, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='0.5', label='Random', linewidth=1.5)
    hi, hi2 = histogram_intersection(counts1, counts2)
    print(hi, hi2)
    r2 = obs_pred_rsquare(counts2, counts1)
    print('CLABSI, p-val:', np.rint(hi), ' r2:', np.rint(r2))
    
    plt.text(0.45*max_x, 0.9*max([max(counts1), max(counts2)]), '∩ = ' + str(int(np.rint(hi))) + '%', fontsize=fs+3)
    plt.ylim(0, 1.1*max([max(counts1), max(counts2)]))
    plt.ylabel('No. of hospitals', fontsize=fs)
    plt.xlabel('SIR, CLABSI', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-3)
    ax2.set_xticks([0, 1, 2]) # choose which x locations to have ticks
    ax2.set_xticklabels([0, 1, 2**2]) # set the labels to display at those ticks
    
    hi_CLABSI.append(hi)
    
    ################################## SUBPLOT 3 ############################################

    ax3 = plt.subplot2grid((rows, cols), (1, 0), colspan=1, rowspan=1)
    tdf = MRSA_df.copy(deep=True)
    y = tdf['O/E']**0.5
    y = y.tolist()
    x = tdf[measure]**0.5
    x = x.tolist()
    x = sorted(x)
    y = sorted(y)

    min_x = min([min(x), min(y)])
    max_x = max([max(x), max(y)])

    counts2, bins2, bars2 = plt.hist(y, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='k', label='Actual', linewidth=2)
    counts1, bins1, bars1 = plt.hist(x, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='0.5', label='Random', linewidth=1.5)
    hi, hi2 = histogram_intersection(counts1, counts2)
    print(hi, hi2)
    r2 = obs_pred_rsquare(counts2, counts1)
    print('MRSA, p-val:', np.rint(hi), ' r2:', np.rint(r2))
    
    plt.text(0.45*max_x, 0.85*max([max(counts1), max(counts2)]), '∩ = ' + str(int(np.rint(hi))) + '%', fontsize=fs+3)
    plt.ylabel('No. of hospitals', fontsize=fs)
    plt.xlabel('SIR, MRSA', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-3)
    ax3.set_xticks([0, 1, 2]) # choose which x locations to have ticks
    ax3.set_xticklabels([0, 1, 2**2]) # set the labels to display at those ticks
    
    hi_MRSA.append(hi)
    
    ################################## SUBPLOT 4 ############################################

    ax4 = plt.subplot2grid((rows, cols), (1, 1), colspan=1, rowspan=1)
    tdf = CDIFF_df.copy(deep=True)
    y = tdf['O/E']**0.5
    y = y.tolist()
    x = tdf[measure]**0.5
    x = x.tolist()
    x = sorted(x)
    y = sorted(y)

    min_x = min([min(x), min(y)])
    max_x = max([max(x), max(y)])
    
    counts2, bins2, bars2 = plt.hist(y, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='k', label='Actual', linewidth=2)
    counts1, bins1, bars1 = plt.hist(x, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='0.5', label='Random', linewidth=1.5)
    
    hi, hi2 = histogram_intersection(counts1, counts2)
    print(hi, hi2)
    r2 = obs_pred_rsquare(counts2, counts1)
    print('CDIFF, p-val:', np.rint(hi), ' r2:', np.rint(r2))
    
    plt.text(0.45*max_x, 410, '∩ = ' + str(int(np.rint(hi))) + '%', fontsize=fs+3)
    plt.ylim(0, 500)
    plt.ylabel('No. of hospitals', fontsize=fs)
    plt.xlabel('SIR, CDIFF', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-3)
    ax4.set_xticks([0, 1, 2]) # choose which x locations to have ticks
    ax4.set_xticklabels([0, 1, 2**2]) # set the labels to display at those ticks
    
    hi_CDIFF.append(hi)
    
    #########################################################################################
    ################################ FINAL FORMATTING #######################################
    #########################################################################################

    plt.subplots_adjust(wspace=0.4, hspace=0.35)
    plt.savefig(mydir+'/7_figures/Fig2/Fig2_' + fdate + '.eps', dpi=400, bbox_inches = "tight")
    plt.close()
    

print('CAUTI, avg histogram intersection:', np.nanmean(hi_CAUTI), ', ±', np.nanstd(hi_CAUTI))

print('CLABSI, avg histogram intersection:', np.nanmean(hi_CLABSI), ', ±', np.nanstd(hi_CLABSI))

print('CDIFF, avg histogram intersection:', np.nanmean(hi_CDIFF), ', ±', np.nanstd(hi_CDIFF))

print('MRSA, avg histogram intersection:', np.nanmean(hi_MRSA), ', ±', np.nanstd(hi_MRSA))
