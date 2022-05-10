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
    
def count_pts_within_radius(x, y, radius, scale=0):
    """Count the number of points within a fixed radius in 2D space"""
    #TODO: see if we can improve performance using KDTree.query_ball_point
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html
    #instead of doing the subset based on the circle
    raw_data = np.array([x, y])
    x = np.array(x)
    y = np.array(y)
    raw_data = raw_data.transpose()
    
    # Get unique data points by adding each pair of points to a set
    unique_points = set()
    for xval, yval in raw_data:
        unique_points.add((xval, yval))
    
    count_data = []
    for a, b in unique_points:
        if scale == 'sqrt':
            num_neighbors = len(x[((sqrt(x) - sqrt(a)) ** 2 +
                                   (sqrt(y) - sqrt(b)) ** 2) <= sqrt(radius) ** 2])
        else:        
            num_neighbors = len(x[((x - a) ** 2 + (y - b) ** 2) <= radius ** 2])
        count_data.append((a, b, num_neighbors))
    return count_data



def plot_color_by_pt_dens(x, y, radius, scale=0, plot_obj=None):
    """Plot bivariate relationships with large n using color for point density

    Inputs:
    x & y -- variables to be plotted
    radius -- the linear distance within which to count points as neighbors
    scale -- a flag to indicate the use of a scale plot (scale = 1)

    The color of each point in the plot is determined by the logarithm (base 10)
    of the number of points that occur with a given radius of the focal point,
    with hotter colors indicating more points. The number of neighboring points
    is determined in linear space regardless of whether a scale plot is
    presented.
    """
    plot_data = count_pts_within_radius(x, y, radius, scale)
    sorted_plot_data = np.array(sorted(plot_data, key=lambda point: point[2]))

    if plot_obj == None:
        plot_obj = plt.axes()
        
    plot_obj.scatter(np.sqrt(sorted_plot_data[:, 0]),
            np.sqrt(sorted_plot_data[:, 1]),
            facecolors='none',
            s = 30, edgecolors='0.1', linewidths=1., #cmap='Greys'
            )
    # plot points
    c = np.array(sorted_plot_data[:, 2])**0.25
    c = np.max(c) - c
    plot_obj.scatter(np.sqrt(sorted_plot_data[:, 0]),
                    np.sqrt(sorted_plot_data[:, 1]),
                    c = c,
                    s = 30, edgecolors='k', linewidths=0.0, cmap='Greys_r',
                    #alpha = 0.5,
                    )
        
    return plot_obj


#########################################################################################
########################## IMPORT HAI DATA ##############################################
#########################################################################################

num_bins = 40

hi_CAUTI = []
hi_CLABSI = []
hi_CDIFF = []
hi_MRSA = []

fdates = ['2014-07-17', '2014-10-23', '2014-12-18', '2015-01-22', '2015-04-16', '2015-05-06', '2015-07-16', '2015-10-08', '2015-12-10', '2016-05-04', '2016-08-10', '2016-11-10', '2017-10-24', '2018-01-26', '2018-05-23', '2018-07-25', '2018-10-31', '2019-03-21', '2019-04-24', '2019-07-02', '2019-10-30', '2020-01-29', '2020-04-22']

for fdate in sorted(fdates, reverse=False):
    print(fdate)
    
    CAUTI_df = pd.read_pickle(mydir + "data/optimized_by_quarter/CAUTI/CAUTI_Data_opt_for_SIRs_" + fdate + ".pkl")
    CAUTI_df = CAUTI_df[CAUTI_df['CAUTI Predicted Cases'] >= 1]
    CAUTI_df = CAUTI_df[CAUTI_df['CAUTI Urinary Catheter Days'] > 0]
    CAUTI_df = CAUTI_df[CAUTI_df['simulated O/E'] >= 0]
    CAUTI_df = CAUTI_df[CAUTI_df['O/E'] >= 0]

    CLABSI_df = pd.read_pickle(mydir + "data/optimized_by_quarter/CLABSI/CLABSI_Data_opt_for_SIRs_" + fdate + ".pkl")
    CLABSI_df = CLABSI_df[CLABSI_df['CLABSI Predicted Cases'] >= 1]
    CLABSI_df = CLABSI_df[CLABSI_df['CLABSI Number of Device Days'] > 0]
    CLABSI_df = CLABSI_df[CLABSI_df['simulated O/E'] >= 0]
    CLABSI_df = CLABSI_df[CLABSI_df['O/E'] >= 0]

    MRSA_df = pd.read_pickle(mydir + "data/optimized_by_quarter/MRSA/MRSA_Data_opt_for_SIRs_" + fdate + ".pkl")
    MRSA_df = MRSA_df[MRSA_df['MRSA Predicted Cases'] >= 1]
    MRSA_df = MRSA_df[MRSA_df['MRSA patient days'] > 0]
    MRSA_df = MRSA_df[MRSA_df['simulated O/E'] >= 0]
    MRSA_df = MRSA_df[MRSA_df['O/E'] >= 0]

    CDIFF_df = pd.read_pickle(mydir + "data/optimized_by_quarter/CDIFF/CDIFF_Data_opt_for_SIRs_" + fdate + ".pkl")
    CDIFF_df = CDIFF_df[CDIFF_df['CDIFF Predicted Cases'] >= 1]
    CDIFF_df = CDIFF_df[CDIFF_df['CDIFF patient days'] > 0]
    CDIFF_df = CDIFF_df[CDIFF_df['simulated O/E'] >= 0]
    CDIFF_df = CDIFF_df[CDIFF_df['O/E'] >= 0]

    #########################################################################################
    ##################### DECLARE FIGURE 3A OBJECT ##########################################
    #########################################################################################

    fig = plt.figure(figsize=(10, 12))
    rows, cols = 4, 3
    fs = 14
    radius = 2

    #########################################################################################
    ################################ GENERATE FIGURE ########################################
    #########################################################################################

    ################################## SUBPLOT 1 ############################################
    tdf = CAUTI_df.copy(deep=True)
    x = tdf['CAUTI Urinary Catheter Days'].tolist()
    y1 = tdf['O/E'].tolist()
    y2 = tdf['simulated O/E'].tolist()
    
    maxx = np.sqrt(np.max(x))
    maxy = 1.1*np.sqrt(max([np.max(y1), np.max(y2)]))
    
    ax1 = plt.subplot2grid((rows, cols), (0, 0), colspan=1, rowspan=1)
    plot_color_by_pt_dens(x, y1, radius, scale='sqrt', plot_obj=ax1)
    plt.ylabel(r'$\sqrt{SIR}$', fontsize=fs)
    plt.xlabel(r'$\sqrt{device\ days}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)
    plt.ylim(-0.2, maxy)
    plt.xlim(-10, maxx)
    plt.hlines(1, -10, 400, colors='k')
    plt.title('Actual SIRs\n', fontsize=fs+3, fontweight='bold', pad=20)
    plt.text(-150, 0.6, 'CAUTI', fontsize=fs+3, fontweight='bold', rotation=90)
    
    ################################## SUBPLOT 2 ############################################

    ax2 = plt.subplot2grid((rows, cols), (0, 1), colspan=1, rowspan=1)
    plot_color_by_pt_dens(x, y2, radius, scale='sqrt', plot_obj=ax2)
    plt.ylabel(r'$\sqrt{SIR}$', fontsize=fs)
    plt.xlabel(r'$\sqrt{device\ days}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)
    plt.ylim(-0.2, maxy)
    plt.xlim(-10, maxx)
    plt.hlines(1, -10, 400, colors='k')
    plt.title('SIRs based on\nrandom sampling', fontsize=fs+3, fontweight='bold', pad=20)

    ################################## SUBPLOT 3 ############################################

    ax3 = plt.subplot2grid((rows, cols), (0, 2), colspan=1, rowspan=1)

    y = tdf['O/E']**0.5
    y = y.tolist()
    x = tdf['simulated O/E']**0.5
    x = x.tolist()
    x = sorted(x)
    y = sorted(y)

    min_x = min([min(x), min(y)])
    max_x = max([max(x), max(y)])

    counts2, bins2, bars2 = plt.hist(y, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='k', label='Actual SIRs', linewidth=1.5)
    counts1, bins1, bars1 = plt.hist(x, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='0.5', label='Random sampling', linewidth=1)
    hi, hi2 = histogram_intersection(counts1, counts2)
    print(hi, hi2)
    r2 = obs_pred_rsquare(counts2, counts1)
    print('CAUTI, p-val:', np.rint(hi), ' r2:', np.rint(r2))
    
    plt.text(0.4*max_x, 0.85*max([max(counts1), max(counts2)]), '∩ = ' + str(int(np.rint(hi))) + '%', fontsize=fs+3)
    #if len(hi_CAUTI) > 1:
    #    hi_avg = str(int(np.rint(np.nanmean(hi_CAUTI))))
    #    hi_std = str(int(np.rint(np.nanstd(hi_CAUTI))))
    #    plt.text(0.15*max_x, 0.75*max([max(counts1), max(counts2)]), r'$\overline{∩}$' + ' = ' + hi_avg + '% ± ' + hi_std, fontsize=fs)
    
    plt.text(0.15, 1.41*max([max(counts1), max(counts2)]), 'Actual SIRs', fontsize=fs+3, color='k',
        fontweight='bold')
    plt.text(-0.35, 1.23*max([max(counts1), max(counts2)]), 'Random sampling', fontsize=fs+3, color='0.5', fontweight='bold')
    
    #plt.legend(loc=0, frameon=False, fontsize=fs)
    #plt.legend(bbox_to_anchor=(-0.3, 1.15, 1.18, .2), loc=10, ncol=1, frameon=False, mode="expand",prop={'size':fs+3, 'weight':'bold', 'color':'r'})
    
    plt.ylabel('No. of hospitals', fontsize=fs)
    plt.xlabel(r'$\sqrt{SIR}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-3)
    hi_CAUTI.append(hi)
    
    ################################## SUBPLOT 4 ############################################
    tdf = CLABSI_df.copy(deep=True)
    x = tdf['CLABSI Number of Device Days'].tolist()
    y1 = tdf['O/E'].tolist()
    y2 = tdf['simulated O/E'].tolist()
    
    maxx = np.sqrt(np.max(x))
    maxy = 1.1*np.sqrt(max([np.max(y1), np.max(y2)]))
    
    ax4 = plt.subplot2grid((rows, cols), (1, 0), colspan=1, rowspan=1)
    plot_color_by_pt_dens(x, y1, radius, scale='sqrt', plot_obj=ax4)
    plt.ylabel(r'$\sqrt{SIR}$', fontsize=fs)
    plt.xlabel(r'$\sqrt{device\ days}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)
    plt.ylim(-0.2, maxy)
    plt.xlim(-10, maxx)
    plt.hlines(1, -10, 400, colors='k')
    plt.text(-170, 0.5, 'CLABSI', fontsize=fs+3, fontweight='bold', rotation=90)
    
    ################################## SUBPLOT 5 ############################################

    ax5 = plt.subplot2grid((rows, cols), (1, 1), colspan=1, rowspan=1)
    plot_color_by_pt_dens(x, y2, radius, scale='sqrt', plot_obj=ax5)
    plt.ylabel(r'$\sqrt{SIR}$', fontsize=fs)
    plt.xlabel(r'$\sqrt{device\ days}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)
    plt.ylim(-0.2, maxy)
    plt.xlim(-10, maxx)
    plt.hlines(1, -10, 400, colors='k')

    ################################## SUBPLOT 6 ############################################

    ax6 = plt.subplot2grid((rows, cols), (1, 2), colspan=1, rowspan=1)

    y = tdf['O/E']**0.5
    y = y.tolist()
    x = tdf['simulated O/E']**0.5
    x = x.tolist()
    x = sorted(x)
    y = sorted(y)

    min_x = min([min(x), min(y)])
    max_x = max([max(x), max(y)])

    counts2, bins2, bars2 = plt.hist(y, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='k', label='Actual', linewidth=1.5)
    counts1, bins1, bars1 = plt.hist(x, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='0.5', label='Random', linewidth=1)
    hi, hi2 = histogram_intersection(counts1, counts2)
    print(hi, hi2)
    r2 = obs_pred_rsquare(counts2, counts1)
    print('CLABSI, p-val:', np.rint(hi), ' r2:', np.rint(r2))
    
    plt.text(0.4*max_x, 0.9*max([max(counts1), max(counts2)]), '∩ = ' + str(int(np.rint(hi))) + '%', fontsize=fs+3)
    #if len(hi_CLABSI) > 1:
    #    hi_avg = str(int(np.rint(np.nanmean(hi_CLABSI))))
    #    hi_std = str(int(np.rint(np.nanstd(hi_CLABSI))))
    #    plt.text(0.15*max_x, 0.78*max([max(counts1), max(counts2)]), r'$\overline{∩}$' + ' = ' + hi_avg + '% ± ' + hi_std, fontsize=fs)
    plt.ylim(0, 1.1*max([max(counts1), max(counts2)]))
    #plt.legend(loc=0, frameon=False, fontsize=fs)
    plt.ylabel('No. of hospitals', fontsize=fs)
    plt.xlabel(r'$\sqrt{SIR}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-3)
    hi_CLABSI.append(hi)
    
    ################################## SUBPLOT 7 ############################################
    tdf = MRSA_df.copy(deep=True)
    x = tdf['MRSA patient days'].tolist()
    y1 = tdf['O/E'].tolist()
    y2 = tdf['simulated O/E'].tolist()
    
    maxx = np.sqrt(np.max(x))
    maxy = 1.1*np.sqrt(max([np.max(y1), np.max(y2)]))
    
    ax7 = plt.subplot2grid((rows, cols), (2, 0), colspan=1, rowspan=1)
    x = tdf['MRSA patient days'].tolist()
    y = tdf['O/E'].tolist()
    plot_color_by_pt_dens(x, y1, radius, scale='sqrt', plot_obj=ax7)
    plt.ylabel(r'$\sqrt{SIR}$', fontsize=fs)
    plt.xlabel(r'$\sqrt{patient\ days}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)
    plt.ylim(-0.2, maxy)
    plt.xlim(-20, maxx)
    plt.hlines(1, -10, 900, colors='k')
    plt.text(-410, 0.6, 'MRSA', fontsize=fs+3, fontweight='bold', rotation=90)
    
    ################################## SUBPLOT 8 ############################################

    ax8 = plt.subplot2grid((rows, cols), (2, 1), colspan=1, rowspan=1)
    x = tdf['MRSA patient days'].tolist()
    y = tdf['simulated O/E'].tolist()
    plot_color_by_pt_dens(x, y2, radius, scale='sqrt', plot_obj=ax8)
    plt.ylabel(r'$\sqrt{SIR}$', fontsize=fs)
    plt.xlabel(r'$\sqrt{patient\ days}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)
    plt.ylim(-0.2, maxy)
    plt.xlim(-20, maxx)
    plt.hlines(1, -10, 900, colors='k')

    ################################## SUBPLOT 9 ############################################

    ax9 = plt.subplot2grid((rows, cols), (2, 2), colspan=1, rowspan=1)

    y = tdf['O/E']**0.5
    y = y.tolist()
    x = tdf['simulated O/E']**0.5
    x = x.tolist()
    x = sorted(x)
    y = sorted(y)

    min_x = min([min(x), min(y)])
    max_x = max([max(x), max(y)])

    counts2, bins2, bars2 = plt.hist(y, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='k', label='Actual', linewidth=1.5)
    counts1, bins1, bars1 = plt.hist(x, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='0.5', label='Random', linewidth=1)
    hi, hi2 = histogram_intersection(counts1, counts2)
    print(hi, hi2)
    r2 = obs_pred_rsquare(counts2, counts1)
    print('MRSA, p-val:', np.rint(hi), ' r2:', np.rint(r2))
    
    plt.text(0.4*max_x, 0.85*max([max(counts1), max(counts2)]), '∩ = ' + str(int(np.rint(hi))) + '%', fontsize=fs+3)
    #if len(hi_MRSA) > 1:
    #    hi_avg = str(int(np.rint(np.nanmean(hi_MRSA))))
    #    hi_std = str(int(np.rint(np.nanstd(hi_MRSA))))
    #    plt.text(0.15*max_x, 0.75*max([max(counts1), max(counts2)]), r'$\overline{∩}$' + ' = ' + hi_avg + '% ± ' + hi_std, fontsize=fs)
    
    #plt.legend(loc=0, frameon=False, fontsize=fs)
    plt.ylabel('No. of hospitals', fontsize=fs)
    plt.xlabel(r'$\sqrt{SIR}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-3)
    hi_MRSA.append(hi)
    
    ################################## SUBPLOT 10 ############################################
    tdf = CDIFF_df.copy(deep=True)
    x = tdf['CDIFF patient days'].tolist()
    y1 = tdf['O/E'].tolist()
    y2 = tdf['simulated O/E'].tolist()
    
    maxx = np.sqrt(np.max(x))
    maxy = 1.1*np.sqrt(max([np.max(y1), np.max(y2)]))
    
    ax10 = plt.subplot2grid((rows, cols), (3, 0), colspan=1, rowspan=1)
    x = tdf['CDIFF patient days'].tolist()
    y = tdf['O/E'].tolist()
    plot_color_by_pt_dens(x, y1, radius, scale='sqrt', plot_obj=ax10)
    plt.ylabel(r'$\sqrt{SIR}$', fontsize=fs)
    plt.xlabel(r'$\sqrt{patient\ days}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)
    plt.ylim(-0.2, maxy)
    plt.xlim(-20, maxx)
    plt.hlines(1, -10, 900, colors='k')
    plt.text(-380, 0.7, 'CDIFF', fontsize=fs+3, fontweight='bold', rotation=90)
    
    ################################## SUBPLOT 11 ############################################

    ax11 = plt.subplot2grid((rows, cols), (3, 1), colspan=1, rowspan=1)
    x = tdf['CDIFF patient days'].tolist()
    y = tdf['simulated O/E'].tolist()
    plot_color_by_pt_dens(x, y2, radius, scale='sqrt', plot_obj=ax11)
    plt.ylabel(r'$\sqrt{SIR}$', fontsize=fs)
    plt.xlabel(r'$\sqrt{patient\ days}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)
    plt.ylim(-0.2, maxy)
    plt.xlim(-20, maxx)
    plt.hlines(1, -10, 900, colors='k')

    ################################## SUBPLOT 12 ############################################

    ax12 = plt.subplot2grid((rows, cols), (3, 2), colspan=1, rowspan=1)

    y = tdf['O/E']**0.5
    y = y.tolist()
    x = tdf['simulated O/E']**0.5
    x = x.tolist()
    x = sorted(x)
    y = sorted(y)

    min_x = min([min(x), min(y)])
    max_x = max([max(x), max(y)])
    
    counts2, bins2, bars2 = plt.hist(y, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='k', label='Actual', linewidth=1.5)
    counts1, bins1, bars1 = plt.hist(x, bins=np.linspace(min_x, max_x, num_bins), histtype='step', density=False, color='0.5', label='Random', linewidth=1)
    
    hi, hi2 = histogram_intersection(counts1, counts2)
    print(hi, hi2)
    r2 = obs_pred_rsquare(counts2, counts1)
    print('CDIFF, p-val:', np.rint(hi), ' r2:', np.rint(r2))
    
    plt.text(0.4*max_x, 1.25*max([max(counts1), max(counts2)]), '∩ = ' + str(int(np.rint(hi))) + '%', fontsize=fs+3)
    #if len(hi_CDIFF) > 1:
    #    hi_avg = str(int(np.rint(np.nanmean(hi_CDIFF))))
    #    hi_std = str(int(np.rint(np.nanstd(hi_CDIFF))))
    #    plt.text(0.15*max_x, 1.1*max([max(counts1), max(counts2)]), r'$\overline{∩}$' + ' = ' + hi_avg + '% ± ' + hi_std, fontsize=fs)
    plt.ylim(0, 1.55*max([max(counts1), max(counts2)]))
    #plt.legend(loc=0, frameon=False, fontsize=fs)
    plt.ylabel('No. of hospitals', fontsize=fs)
    plt.xlabel(r'$\sqrt{SIR}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-3)
    hi_CDIFF.append(hi)
    
    #########################################################################################
    ################################ FINAL FORMATTING #######################################
    #########################################################################################

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(mydir+'/figures/Fig1/Fig1_' + fdate + '.eps', dpi=300, bbox_inches = "tight")
    plt.close()
    #break
    

print('CAUTI, avg histogram intersection:', np.nanmean(hi_CAUTI), ', ±', np.nanstd(hi_CAUTI))

print('CLABSI, avg histogram intersection:', np.nanmean(hi_CLABSI), ', ±', np.nanstd(hi_CLABSI))

print('CDIFF, avg histogram intersection:', np.nanmean(hi_CDIFF), ', ±', np.nanstd(hi_CDIFF))

print('MRSA, avg histogram intersection:', np.nanmean(hi_MRSA), ', ±', np.nanstd(hi_MRSA))
