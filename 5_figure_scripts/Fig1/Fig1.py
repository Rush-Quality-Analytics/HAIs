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
            s = 30, edgecolors='0.1', linewidths=1.5, #cmap='Greys'
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

fdates = ['2020-04-22']#, '2014-07-17', '2014-10-23', '2014-12-18', '2015-01-22', '2015-04-16', '2015-05-06', '2015-07-16', '2015-10-08', '2015-12-10', '2016-05-04', '2016-08-10', '2016-11-10', '2017-10-24', '2018-01-26', '2018-05-23', '2018-07-25', '2018-10-31', '2019-03-21', '2019-04-24', '2019-07-02', '2019-10-30', '2020-01-29']

for fdate in sorted(fdates, reverse=False):
    print(fdate)
    
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
    fs = 16
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
    plt.ylabel('SIR, CAUTI', fontsize=fs)
    plt.xlabel('Device days', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-3)
    plt.ylim(-0.1, maxy)
    plt.xlim(10, maxx)
    ax1.set_xticks([0, 100, 200, 300]) # choose which x locations to have ticks
    ax1.set_xticklabels([0, 100**2, 200**2, 300**2]) # set the labels to display at those ticks
    ax1.set_yticks([0, 0.5, 1, 1.5, 2.0]) # choose which x locations to have ticks
    ax1.set_yticklabels([0, 0.5**2, 1, 1.5**2, 2.0**2]) # set the labels to display at those ticks
    plt.hlines(1, -10, 400, colors='k')
    
    
    ################################## SUBPLOT 2 ############################################
    tdf = CLABSI_df.copy(deep=True)
    x = tdf['CLABSI Number of Device Days'].tolist()
    y1 = tdf['O/E'].tolist()
    y2 = tdf['simulated O/E'].tolist()
    
    maxx = np.sqrt(np.max(x))
    maxy = 1.1*np.sqrt(max([np.max(y1), np.max(y2)]))
    
    ax2 = plt.subplot2grid((rows, cols), (0, 1), colspan=1, rowspan=1)
    plot_color_by_pt_dens(x, y1, radius, scale='sqrt', plot_obj=ax2)
    plt.ylabel('SIR, CLABSI', fontsize=fs)
    plt.xlabel('Device days', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-3)
    plt.ylim(-0.1, maxy)
    plt.xlim(10, maxx)
    ax2.set_xticks([0, 100, 200, 300]) # choose which x locations to have ticks
    ax2.set_xticklabels([0, 100**2, 200**2, 300**2]) # set the labels to display at those ticks
    ax2.set_yticks([0, 0.5, 1, 1.5, 2.0]) # choose which x locations to have ticks
    ax2.set_yticklabels([0, 0.5**2, 1, 1.5**2, 2.0**2]) # set the labels to display at those ticks
    plt.hlines(1, -10, 400, colors='k')
    

    ################################## SUBPLOT 3 ############################################
    tdf = MRSA_df.copy(deep=True)
    x = tdf['MRSA patient days'].tolist()
    y1 = tdf['O/E'].tolist()
    y2 = tdf['simulated O/E'].tolist()
    
    maxx = np.sqrt(np.max(x))
    minx = np.sqrt(np.min(x))
    maxy = 1.1*np.sqrt(max([np.max(y1), np.max(y2)]))
    
    ax3 = plt.subplot2grid((rows, cols), (1, 0), colspan=1, rowspan=1)
    x = tdf['MRSA patient days'].tolist()
    y = tdf['O/E'].tolist()
    plot_color_by_pt_dens(x, y1, radius, scale='sqrt', plot_obj=ax3)
    plt.ylabel('SIR, MRSA', fontsize=fs)
    plt.xlabel('Patient days', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-3)
    plt.ylim(-0.1, maxy)
    plt.xlim(minx, maxx)
    ax3.set_xticks([80, 250, 500, 750]) # choose which x locations to have ticks
    ax3.set_xticklabels([80**2, 250**2, 500**2, 750**2]) # set the labels to display at those ticks
    ax3.set_yticks([0, 0.5, 1, 1.5, 2.0]) # choose which x locations to have ticks
    ax3.set_yticklabels([0, 0.5**2, 1, 1.5**2, 2.0**2]) # set the labels to display at those ticks
    plt.hlines(1, -10, 900, colors='k')
    
    
    ################################## SUBPLOT 4 ############################################
    tdf = CDIFF_df.copy(deep=True)
    x = tdf['CDIFF patient days'].tolist()
    y1 = tdf['O/E'].tolist()
    y2 = tdf['simulated O/E'].tolist()
    
    maxx = np.sqrt(np.max(x))
    maxy = 1.1*np.sqrt(max([np.max(y1), np.max(y2)]))
    
    ax4 = plt.subplot2grid((rows, cols), (1, 1), colspan=1, rowspan=1)
    x = tdf['CDIFF patient days'].tolist()
    y = tdf['O/E'].tolist()
    plot_color_by_pt_dens(x, y1, radius, scale='sqrt', plot_obj=ax4)
    plt.ylabel('SIR, CDIFF', fontsize=fs)
    plt.xlabel('Patient days', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-3)
    plt.ylim(-0.1, maxy)
    plt.xlim(500, maxx)
    ax4.set_xticks([0, 250, 500, 750]) # choose which x locations to have ticks
    ax4.set_xticklabels([0, 250**2, 500**2, 750**2]) # set the labels to display at those ticks
    ax4.set_yticks([0, 0.5, 1, 1.5, 2.0]) # choose which x locations to have ticks
    ax4.set_yticklabels([0, 0.5**2, 1, 1.5**2, 2.0**2]) # set the labels to display at those ticks
    plt.hlines(1, -10, 900, colors='k')
    
    
    #########################################################################################
    ################################ FINAL FORMATTING #######################################
    #########################################################################################

    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.savefig(mydir+'/7_figures/Fig1/Fig1_' + fdate + '.png', dpi=400, bbox_inches = "tight")
    plt.close()
    #break
    