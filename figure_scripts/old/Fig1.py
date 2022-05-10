import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from math import pi
import sys
import os
import scipy as sc
from scipy import stats # scientific python statistical package
from scipy.optimize import curve_fit # optimization for fitting curves

import warnings
from scipy.stats import binned_statistic
from numpy import log10, sqrt
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.patches as patches
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from scipy.stats import binned_statistic

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
warnings.filterwarnings('ignore')

mydir = os.path.expanduser("~/GitHub/HAIs/")

#########################################################################################
################################ FUNCTIONS ##############################################

def powerlaw(x, a, b, c, d):
    return a/((x)**c + d)

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
            s = 30, edgecolors='0.1', linewidths=0.75, cmap='Greys'
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




#########################################################################################
##################### DECLARE FIGURE OBJECT ##########################################
#########################################################################################

fig = plt.figure(figsize=(10, 10))
rows, cols = 2, 2
fs = 12
kernel = 0.05
radius = 1

#########################################################################################
################################ GENERATE FIGURE ########################################
#########################################################################################

################################## SUBPLOT 1 ############################################

ax1 = plt.subplot2grid((rows, cols), (0, 0), colspan=1, rowspan=1)

x = np.random.logseries(0.97, 1000000)
x = x.tolist()
n = []

obs = []
exp = []
oe = []
p = 0.05
for i, xval in enumerate(x):
    if p*xval >= 1:
        events = np.random.binomial(1, p=p, size=int(xval)).tolist()
        ct = events.count(1)
        obs.append(ct)
        exp.append(xval*p)
        oe.append(ct/(xval*p))
        n.append(xval)

plot_color_by_pt_dens(n, oe, radius, loglog=0, plot_obj=ax1)

plt.ylabel('Observed / Expected', fontsize=fs)
plt.xlabel('Sample size', fontsize=fs)
plt.tick_params(axis='both', labelsize=fs-2)
plt.hlines(1, 0, max(n), colors='k')
plt.xlim(0, max(n)+10)
#plt.ylim(-0.25, max(oe))

'''
inset_axes2 = inset_axes(ax1,
                    width="40%", # width = 30% of parent_bbox
                    height="40%", # height : 1 inch
                    loc=1)

bins = np.linspace(1, max(n), 80).tolist()
bins.append(1011)
ix = []
iy = []
for i, bin in enumerate(bins):
    zeros = 0
    ni = 0
    if bin == max(bins):
        break
    for ii, val in enumerate(n):
        if val >= bin and val < bins[i + 1]:
            if obs[ii] == 0:
                zeros += 1
            ni += 1
    if ni > 0:
        iy.append(100*zeros/ni)
        ix.append(bin)

ix = np.array(ix)
iy = np.array(iy)
plt.scatter(ix, iy, s=5, c='k')

popt, pcov = curve_fit(powerlaw, ix, iy,
                       method='lm', maxfev=50000)
# get predicted y values
#pred_y = powerlaw(ix, *popt)
#print(popt)
#plt.plot(ix, pred_y, c='0.5', linewidth=1)
#plt.xlim(-10, 350)

plt.tick_params(axis='both', labelsize=fs-3)
plt.ylabel('0/E = 0', fontsize=fs-3)
plt.xlabel('Sample size', fontsize=fs-3)
'''

#########################################################################################
################################ FINAL FORMATTING #######################################
#########################################################################################

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig(mydir+'/figures/Fig1.png', dpi=400, bbox_inches = "tight")
plt.close()
