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
from PIL import Image
from io import BytesIO


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
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

def count_pts_within_radius(x, y, radius, logscale=0):
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
        if logscale == 1:
            num_neighbors = len(x[((log10(x) - log10(a)) ** 2 +
                                   (log10(y) - log10(b)) ** 2) <= log10(radius) ** 2])
        else:        
            num_neighbors = len(x[((x - a) ** 2 + (y - b) ** 2) <= radius ** 2])
        count_data.append((a, b, num_neighbors))
    return count_data



def plot_color_by_pt_dens(x, y, radius, loglog=0, plot_obj=None):
    """Plot bivariate relationships with large n using color for point density

    Inputs:
    x & y -- variables to be plotted
    radius -- the linear distance within which to count points as neighbors
    loglog -- a flag to indicate the use of a loglog plot (loglog = 1)

    The color of each point in the plot is determined by the logarithm (base 10)
    of the number of points that occur with a given radius of the focal point,
    with hotter colors indicating more points. The number of neighboring points
    is determined in linear space regardless of whether a loglog plot is
    presented.
    """
    plot_data = count_pts_within_radius(x, y, radius, loglog)
    sorted_plot_data = np.array(sorted(plot_data, key=lambda point: point[2]))

    if plot_obj == None:
        plot_obj = plt.axes()
        
    plot_obj.scatter(sorted_plot_data[:, 0],
            sorted_plot_data[:, 1],
            facecolors='none',
            s = 30, edgecolors='0.1', linewidths=1.0, cmap='Greys_r',
            )
    # plot points
    c = np.array(sorted_plot_data[:, 2])**0.25
    c = np.max(c) - c
    plot_obj.scatter(sorted_plot_data[:, 0],
                    sorted_plot_data[:, 1],
                    c = c,
                    s = 30, edgecolors='k', linewidths=0.0, cmap='Greys_r',
                    #alpha = 0.5,
                    )
        
    return plot_obj




CAUTI_r2s = []
CLABSI_r2s = []
MRSA_r2s = []
CDIFF_r2s = []

fdates = ['2014-07-17', '2014-10-23', '2014-12-18', '2015-01-22', '2015-04-16', '2015-05-06', '2015-07-16', '2015-10-08', '2015-12-10', '2016-05-04', '2016-08-10', '2016-11-10', '2017-10-24', '2018-01-26', '2018-05-23', '2018-07-25', '2018-10-31', '2019-03-21', '2019-04-24', '2019-07-02', '2019-10-30', '2020-01-29', '2020-04-22']

for fdate in fdates:
    #########################################################################################
    ########################## IMPORT HAI DATA ##############################################
    #########################################################################################

    CAUTI_df = pd.read_pickle(mydir + "1_data/optimized_by_quarter/CAUTI/CAUTI_Data_opt_for_SIRs_" + fdate + ".pkl")
    CAUTI_df = CAUTI_df[CAUTI_df['CAUTI Predicted Cases'] >= 1]
    CAUTI_df = CAUTI_df[CAUTI_df['CAUTI Urinary Catheter Days'] > 0]
    CAUTI_df = CAUTI_df[CAUTI_df['simulated O/E'] >= 0]
    CAUTI_df = CAUTI_df[CAUTI_df['O/E'] >= 0]

    CLABSI_df = pd.read_pickle(mydir + "1_data/optimized_by_quarter/CLABSI/CLABSI_Data_opt_for_SIRs_" + fdate + ".pkl")
    CLABSI_df = CLABSI_df[CLABSI_df['CLABSI Predicted Cases'] >= 1]
    CLABSI_df = CLABSI_df[CLABSI_df['CLABSI Number of Device Days'] > 0]
    CLABSI_df = CLABSI_df[CLABSI_df['simulated O/E'] >= 0]
    CLABSI_df = CLABSI_df[CLABSI_df['O/E'] >= 0]

    MRSA_df = pd.read_pickle(mydir + "1_data/optimized_by_quarter/MRSA/MRSA_Data_opt_for_SIRs_" + fdate + ".pkl")
    MRSA_df = MRSA_df[MRSA_df['MRSA Predicted Cases'] >= 1]
    MRSA_df = MRSA_df[MRSA_df['MRSA patient days'] > 0]
    MRSA_df = MRSA_df[MRSA_df['simulated O/E'] >= 0]
    MRSA_df = MRSA_df[MRSA_df['O/E'] >= 0]

    CDIFF_df = pd.read_pickle(mydir + "1_data/optimized_by_quarter/CDIFF/CDIFF_Data_opt_for_SIRs_" + fdate + ".pkl")
    CDIFF_df = CDIFF_df[CDIFF_df['CDIFF Predicted Cases'] >= 1]
    CDIFF_df = CDIFF_df[CDIFF_df['CDIFF patient days'] > 0]
    CDIFF_df = CDIFF_df[CDIFF_df['simulated O/E'] >= 0]
    CDIFF_df = CDIFF_df[CDIFF_df['O/E'] >= 0]

    #########################################################################################
    ##################### DECLARE FIGURE 3A OBJECT ##########################################
    #########################################################################################

    fig = plt.figure(figsize=(8, 8))
    rows, cols = 2, 2
    fs = 14
    radius = 1

    matplotlib.rcParams['legend.handlelength'] = 0
    matplotlib.rcParams['legend.numpoints'] = 1

    #########################################################################################
    ################################ GENERATE FIGURE ########################################
    #########################################################################################

    ################################## SUBPLOT 1 ############################################

    ax1 = plt.subplot2grid((rows, cols), (0, 0), colspan=1, rowspan=1)
    #ax1.set_xticks([0, 2, 4, 6, 8, 10, 12])

    x = CAUTI_df['expected O']**0.5
    y = CAUTI_df['CAUTI Observed Cases']**0.5
    r2 = obs_pred_rsquare(y, x)
    CAUTI_r2s.append(r2)
    
    slope, intercept, r, p, se = sc.stats.linregress(x, y)
    maxv = min([np.max(x), np.max(y)])
    plt.plot([0, maxv], [0, maxv], 'k', linewidth=2)
    
    plot_color_by_pt_dens(x, y, radius, plot_obj=ax1)
    plt.tick_params(axis='both', labelsize=fs-4)
    plt.xlabel('Cases expected at random', fontsize=fs-1)
    plt.ylabel('Reported CAUTI cases', fontsize=fs)
    plt.text(0.0*max(x), 0.9*max(y), r'$\overline{r^{2}}$' + ' = ' + str(np.round(np.mean(CAUTI_r2s), 2)) + ' ± ' + str(np.round(np.std(CAUTI_r2s), 2)), fontsize=fs-2)
    plt.text(0.0*max(x), 0.8*max(y), r'$r^{2}$' + ' = ' + str(np.round(r2, 2)), fontsize=fs-2)
    ax1.set_xticks([0, 2, 4, 6, 8, 10]) # choose which x locations to have ticks
    ax1.set_xticklabels([0, 2**2, 4**2, 6**2, 8**2, 10**2]) # set the labels to display at those ticks
    ax1.set_yticks([0, 2, 4, 6, 8, 10]) # choose which x locations to have ticks
    ax1.set_yticklabels([0, 2**2, 4**2, 6**2, 8**2, 10**2]) # set the labels to display at those ticks
    
    ################################## SUBPLOT 2 ############################################

    ax2 = plt.subplot2grid((rows, cols), (0, 1), colspan=1, rowspan=1)
    x = CLABSI_df['expected O']**0.5
    y = CLABSI_df['CLABSI Observed Cases']**0.5
    r2 = obs_pred_rsquare(y, x)
    CLABSI_r2s.append(r2)
    
    maxv = min([np.max(x), np.max(y)])
    plt.plot([0, maxv], [0, maxv], 'k', linewidth=2)
    
    plot_color_by_pt_dens(x, y, radius, plot_obj=ax2)
    plt.tick_params(axis='both', labelsize=fs-4)
    plt.xlabel('Cases expected at random', fontsize=fs-1)
    plt.ylabel('Reported CLABSI cases', fontsize=fs)
    plt.text(0.0*max(x), 0.9*max(y), r'$\overline{r^{2}}$' + ' = ' + str(np.round(np.mean(CLABSI_r2s), 2)) + ' ± ' + str(np.round(np.std(CLABSI_r2s), 2)), fontsize=fs-2)
    plt.text(0.0*max(x), 0.8*max(y), r'$r^{2}$' + ' = ' + str(np.round(r2, 2)), fontsize=fs-2)
    ax2.set_xticks([0, 2, 4, 6, 8, 10]) # choose which x locations to have ticks
    ax2.set_xticklabels([0, 2**2, 4**2, 6**2, 8**2, 10**2]) # set the labels to display at those ticks
    ax2.set_yticks([0, 2, 4, 6, 8, 10, 12]) # choose which x locations to have ticks
    ax2.set_yticklabels([0, 2**2, 4**2, 6**2, 8**2, 10**2, 12**2]) # set the labels to display at those ticks
    
    
    ################################## SUBPLOT 3 ############################################

    ax3 = plt.subplot2grid((rows, cols), (1, 0), colspan=1, rowspan=1)

    x = MRSA_df['expected O']**0.5
    y = MRSA_df['MRSA Observed Cases']**0.5
    r2 = obs_pred_rsquare(y, x)
    MRSA_r2s.append(r2)
    
    slope, intercept, r, p, se = sc.stats.linregress(x, y)
    maxv = min([np.max(x), np.max(y)])
    plt.plot([0, maxv], [0, maxv], 'k', linewidth=2)
    
    plot_color_by_pt_dens(x, y, radius, plot_obj=ax3)
    plt.tick_params(axis='both', labelsize=fs-4)
    plt.xlabel('Cases expected at random', fontsize=fs-1)
    plt.ylabel('Reported MRSA cases', fontsize=fs)
    plt.text(0.0*max(x), 0.9*max(y), r'$\overline{r^{2}}$' + ' = ' + str(np.round(np.mean(MRSA_r2s), 2)) + ' ± ' + str(np.round(np.std(MRSA_r2s), 2)), fontsize=fs-2)
    plt.text(0.0*max(x), 0.8*max(y), r'$r^{2}$' + ' = ' + str(np.round(r2, 2)), fontsize=fs-2)
    ax3.set_xticks([0, 2, 4, 6, 8,]) # choose which x locations to have ticks
    ax3.set_xticklabels([0, 2**2, 4**2, 6**2, 8**2]) # set the labels to display at those ticks
    ax3.set_yticks([0, 1, 2, 3, 4, 5, 6, 7]) # choose which x locations to have ticks
    ax3.set_yticklabels([0, 1**2, 2**2, 3**2, 4**2, 5**2, 6**2, 7**2]) # set the labels to display at those ticks
    
    ################################## SUBPLOT 4 ############################################

    ax4 = plt.subplot2grid((rows, cols), (1, 1), colspan=1, rowspan=1)

    x = CDIFF_df['expected O']**0.5
    y = CDIFF_df['CDIFF Observed Cases']**0.5
    r2 = obs_pred_rsquare(y, x)
    CDIFF_r2s.append(r2)
    
    slope, intercept, r, p, se = sc.stats.linregress(x, y)
    maxv = min([np.max(x), np.max(y)])
    plt.plot([0, maxv], [0, maxv], 'k', linewidth=2)
    
    plot_color_by_pt_dens(x, y, radius, plot_obj=ax4)
    plt.tick_params(axis='both', labelsize=fs-4)
    plt.xlabel('Cases expected at random', fontsize=fs-1)
    plt.ylabel('Reported CDIFF cases', fontsize=fs)
    plt.text(0.0*max(x), 0.9*max(y), r'$\overline{r^{2}}$' + ' = ' + str(np.round(np.mean(CDIFF_r2s), 2)) + ' ± ' + str(np.round(np.std(CDIFF_r2s), 2)), fontsize=fs-2)
    plt.text(0.0*max(x), 0.8*max(y), r'$r^{2}$' + ' = ' + str(np.round(r2, 2)), fontsize=fs-2)
    ax4.set_xticks([0, 5, 10, 15]) # choose which x locations to have ticks
    ax4.set_xticklabels([0, 5**2, 10**2, 15**2]) # set the labels to display at those ticks
    ax4.set_yticks([0, 4, 8, 12, 16, 20]) # choose which x locations to have ticks
    ax4.set_yticklabels([0, 4**2, 8**2, 12**2, 16**2, 20**2]) # set the labels to display at those ticks
    
    #########################################################################################
    ################################ FINAL FORMATTING #######################################
    #########################################################################################
    if fdate == '2020-04-22':
        plt.subplots_adjust(wspace=0.35, hspace=0.3)
        fig.savefig(mydir+'/7_figures/Fig2/Fig2_' + fdate + '.eps', dpi=300, format="eps", bbox_inches = "tight")

        '''
        # save figure
        # (1) save the image in memory in PNG format
        f1 = BytesIO()
        fig.savefig(f1, format='png', dpi=300, bbox_inches = "tight")

        # (2) load this image into PIL
        f2 = Image.open(f1)

        # (3) save as TIFF, EPS, etc
        f2.save(mydir+'/figures/Fig2/Fig2_' + fdate + '.tiff')
        f1.close()
        '''
    
print('cauti:', np.mean(CAUTI_r2s), np.std(CAUTI_r2s))
print('clabsi:', np.mean(CLABSI_r2s), np.std(CLABSI_r2s))
print('mrsa:', np.mean(MRSA_r2s), np.std(MRSA_r2s))
print('cdiff:', np.mean(CDIFF_r2s), np.std(CDIFF_r2s))

