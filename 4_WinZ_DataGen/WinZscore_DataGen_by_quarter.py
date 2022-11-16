import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from math import pi
import sys
import os
import scipy as sc
from scipy import special
import warnings
from scipy.stats import binned_statistic
from numpy import log10, sqrt
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.patches as patches
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from scipy.stats.kde import gaussian_kde
from scipy.stats.mstats import winsorize
from statsmodels.stats.proportion import proportion_confint

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

ci_alpha = 0.4
b_method = 'normal'
metrics = ['SIS']

def Winsorize_it(x):
    X = []
    for i in x:
        if np.isnan(i) == False:
            X.append(i)
        
    p5 = np.percentile(np.array(X), 5)
    p95 = np.percentile(np.array(X), 95)
    
    WinScores = []
    for i in x:
        if np.isnan(i) == False: 
            if i >= p5 and i <= p95:
                WinScores.append(i)
            elif i < p5:
                WinScores.append(p5)
            elif i > p95:
                WinScores.append(p95)
        else:
            WinScores.append(np.nan)
            
    return WinScores
        

def ZScore_it(x):
    zscores = stats.zscore(x, nan_policy='omit')
    return zscores


def SIS(O, E, P, uP, uE, V):
    
    sis = (O - E) / P
    
    return sis
    

def get_btr(O, E, uE, P):
    O = O.tolist()
    E = E.tolist()
    uE = uE.tolist()
    P = P.tolist()
    btr = []
    
    for i, exp_val in enumerate(E):
        obs_val = O[i]
        pred_val = P[i]
        if obs_val < exp_val and obs_val < pred_val:
            btr.append(1)
        else:
            btr.append(0)
    
    if len(list(set(btr))) < 2:
        print('btr set:', list(set(btr)))
    return btr
    
    
    
for metric in metrics:

    mydir = os.path.expanduser("~/GitHub/HAIs/")

    mrsa_df, cauti_df, clabsi_df, cdiff_df = 0,0,0,0

    fdates = ['2014-07-17', '2014-10-23', '2014-12-18', '2015-01-22', '2015-04-16', '2015-05-06', '2015-07-16', '2015-10-08', '2015-12-10', '2016-05-04', '2016-08-10', '2016-11-10', '2017-10-24', '2018-01-26', '2018-05-23', '2018-07-25', '2018-10-31', '2019-03-21', '2019-04-24', '2019-07-02', '2019-10-30', '2020-01-29', '2020-04-22']

    for fdate in fdates:

        #########################################################################################
        ########################## IMPORT HAI DATA ##############################################


        ################################################################
        ################### MRSA #######################################
        ################################################################
        
        tdf = pd.read_pickle(mydir + "1_data/optimized_by_quarter/MRSA/MRSA_Data_opt_for_SIRs_" + fdate + ".pkl")
        #print(list(tdf))
        #print(tdf.head())
        
        tdf.drop_duplicates(inplace=True)
        
        tdf.rename(columns={
                            'MRSA Predicted Cases': 'Predicted Cases',
                            'MRSA patient days': 'Days',
                            'MRSA': 'SIR',
                            'MRSA upper CL': 'upper CL',
                            'MRSA lower CL': 'lower CL',
                            'MRSA Observed Cases': 'Observed Cases',
                            }, inplace=True)
        tdf = tdf[tdf['Predicted Cases'] >= 1]
        tdf['HAI'] = 'MRSA'
        tdf['upper CL'] = tdf['upper CL'].astype(float)
        
        days = np.array(tdf['Days'])
        pi = np.array(tdf['pi_opt'])
        z = np.array(tdf['z_opt'])
        p = (pi * days / (days + z))
        ci_low, ci_upp = proportion_confint(tdf['expected O'], days, alpha=ci_alpha, method=b_method)
        tdf['exp_random'] = days * ci_upp
        tdf['SISc'] = (tdf['Observed Cases'] - tdf['exp_random']) / (tdf['Predicted Cases'] - tdf['exp_random'])
        tdf['exp_random'] = days * p
        tdf['SISo'] = (tdf['Observed Cases'] - tdf['exp_random']) / (tdf['Predicted Cases'])
        
        tdf['exp_random'] = days * ci_upp
        tdf['SIS'] = SIS(tdf['Observed Cases'], 
                         tdf['expected O'], 
                         tdf['Predicted Cases'], 
                         tdf['upper CL'], 
                         tdf['exp_random'],
                         tdf['Days'])
        
        btr = get_btr(tdf['Observed Cases'], tdf['expected O'], tdf['exp_random'], tdf['Predicted Cases'])
        tdf['Observed is better than random'] = btr
        tdf['Winzorized SIR'] = Winsorize_it(tdf['SIR'].tolist())
        tdf['Winzorized SIS'] = Winsorize_it(tdf['SIS'].tolist())
        tdf['Winzorized SISc'] = Winsorize_it(tdf['SISc'].tolist())
        tdf['Winzorized SISo'] = Winsorize_it(tdf['SISo'].tolist())
        tdf['Winzorized z SIR'] = ZScore_it(tdf['Winzorized SIR'])
        tdf['Winzorized z SIS'] = ZScore_it(tdf['Winzorized SIS'])
        tdf['Winzorized z SISc'] = ZScore_it(tdf['Winzorized SISc'])
        tdf['Winzorized z SISo'] = ZScore_it(tdf['Winzorized SISo'])
        
        
        tdf['Winzorized z SISo rank'] = tdf['Winzorized z SISo'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        tdf['Winzorized z SIS rank'] = tdf['Winzorized z SIS'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        tdf['Winzorized z SISc rank'] = tdf['Winzorized z SISc'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        tdf['Winzorized z SIR rank'] = tdf['Winzorized z SIR'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        
        if fdate == '2014-07-17':
            mrsa_df = tdf.copy(deep=True)
        else:
            mrsa_df = pd.concat([mrsa_df, tdf], ignore_index=True)


        ################################################################
        ################## CDIFF #######################################
        ################################################################
        
        tdf = pd.read_pickle(mydir + "1_data/optimized_by_quarter/CDIFF/CDIFF_Data_opt_for_SIRs_" + fdate + ".pkl")
        tdf.drop_duplicates(inplace=True)
        
        tdf.rename(columns={
                            'CDIFF Predicted Cases': 'Predicted Cases',
                            'CDIFF patient days': 'Days',
                            'CDIFF': 'SIR',
                            'CDIFF upper CL': 'upper CL',
                            'CDIFF lower CL': 'lower CL',
                            'CDIFF Observed Cases': 'Observed Cases',
                            }, inplace=True)
        tdf = tdf[tdf['Predicted Cases'] >= 1]
        tdf['HAI'] = 'CDIFF'
        tdf['upper CL'] = tdf['upper CL'].astype(float)
        
        days = np.array(tdf['Days'])
        pi = np.array(tdf['pi_opt'])
        z = np.array(tdf['z_opt'])
        p = (pi * days / (days + z))
        ci_low, ci_upp = proportion_confint(tdf['expected O'], days, alpha=ci_alpha, method=b_method)
        tdf['exp_random'] = days * ci_upp
        tdf['SISc'] = (tdf['Observed Cases'] - tdf['exp_random']) / (tdf['Predicted Cases'] - tdf['exp_random'])
        tdf['exp_random'] = days * p
        tdf['SISo'] = (tdf['Observed Cases'] - tdf['exp_random']) / (tdf['Predicted Cases'])
        
        tdf['exp_random'] = days * ci_upp
        tdf['SIS'] = SIS(tdf['Observed Cases'], 
                         tdf['expected O'], 
                         tdf['Predicted Cases'], 
                         tdf['upper CL'], 
                         tdf['exp_random'],
                         tdf['Days'])
        
        btr = get_btr(tdf['Observed Cases'], tdf['expected O'], tdf['exp_random'], tdf['Predicted Cases'])
        tdf['Observed is better than random'] = btr
        tdf['Winzorized SIR'] = winsorize(tdf['SIR'], limits=[0.05, 0.05])
        tdf['Winzorized SIS'] = winsorize(tdf['SIS'], limits=[0.05, 0.05])
        tdf['Winzorized SISc'] = winsorize(tdf['SISc'], limits=[0.05, 0.05])
        tdf['Winzorized SISo'] = winsorize(tdf['SISo'], limits=[0.05, 0.05])
        tdf['Winzorized z SIR'] = stats.zscore(tdf['Winzorized SIR'])
        tdf['Winzorized z SIS'] = stats.zscore(tdf['Winzorized SIS'])
        tdf['Winzorized z SISc'] = stats.zscore(tdf['Winzorized SISc'])
        tdf['Winzorized z SISo'] = stats.zscore(tdf['Winzorized SISo'])
        
        tdf['Winzorized z SISo rank'] = tdf['Winzorized z SISo'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        tdf['Winzorized z SIS rank'] = tdf['Winzorized z SIS'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        tdf['Winzorized z SISc rank'] = tdf['Winzorized z SISc'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        tdf['Winzorized z SIR rank'] = tdf['Winzorized z SIR'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        
        if fdate == '2014-07-17':
            cdiff_df = tdf.copy(deep=True)
        else:
            cdiff_df = pd.concat([cdiff_df, tdf], ignore_index=True)


        ################################################################
        ################# CLABSI #######################################
        ################################################################
        
        tdf = pd.read_pickle(mydir + "1_data/optimized_by_quarter/CLABSI/CLABSI_Data_opt_for_SIRs_" + fdate + ".pkl")
        tdf.drop_duplicates(inplace=True)
        
        tdf.rename(columns={
                            'CLABSI Predicted Cases': 'Predicted Cases',
                            'CLABSI Number of Device Days': 'Days',
                            'CLABSI': 'SIR',
                            'CLABSI upper CL': 'upper CL',
                            'CLABSI lower CL': 'lower CL',
                            'CLABSI Observed Cases': 'Observed Cases',
                            }, inplace=True)
        tdf = tdf[tdf['Predicted Cases'] >= 1]
        tdf['HAI'] = 'CLABSI'
        tdf['upper CL'] = tdf['upper CL'].astype(float)
        
        days = np.array(tdf['Days'])
        pi = np.array(tdf['pi_opt'])
        z = np.array(tdf['z_opt'])
        p = (pi * days / (days + z))
        ci_low, ci_upp = proportion_confint(tdf['expected O'], days, alpha=ci_alpha, method=b_method)
        tdf['exp_random'] = days * ci_upp
        tdf['SISc'] = (tdf['Observed Cases'] - tdf['exp_random']) / (tdf['Predicted Cases'] - tdf['exp_random'])
        tdf['exp_random'] = days * p
        tdf['SISo'] = (tdf['Observed Cases'] - tdf['exp_random']) / (tdf['Predicted Cases'])
        
        tdf['exp_random'] = days * ci_upp
        tdf['SIS'] = SIS(tdf['Observed Cases'], 
                         tdf['expected O'], 
                         tdf['Predicted Cases'], 
                         tdf['upper CL'], 
                         tdf['exp_random'],
                         tdf['Days'])
        
        btr = get_btr(tdf['Observed Cases'], tdf['expected O'], tdf['exp_random'], tdf['Predicted Cases'])
        tdf['Observed is better than random'] = btr
        tdf['Winzorized SIR'] = winsorize(tdf['SIR'], limits=[0.05, 0.05])
        tdf['Winzorized SIS'] = winsorize(tdf['SIS'], limits=[0.05, 0.05])
        tdf['Winzorized SISc'] = winsorize(tdf['SISc'], limits=[0.05, 0.05])
        tdf['Winzorized SISo'] = winsorize(tdf['SISo'], limits=[0.05, 0.05])
        tdf['Winzorized z SIR'] = stats.zscore(tdf['Winzorized SIR'])
        tdf['Winzorized z SIS'] = stats.zscore(tdf['Winzorized SIS'])
        tdf['Winzorized z SISc'] = stats.zscore(tdf['Winzorized SISc'])
        tdf['Winzorized z SISo'] = stats.zscore(tdf['Winzorized SISo'])
        
        tdf['Winzorized z SISo rank'] = tdf['Winzorized z SISo'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        tdf['Winzorized z SIS rank'] = tdf['Winzorized z SIS'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        tdf['Winzorized z SISc rank'] = tdf['Winzorized z SISc'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        tdf['Winzorized z SIR rank'] = tdf['Winzorized z SIR'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        
        if fdate == '2014-07-17':
            clabsi_df = tdf.copy(deep=True)
        else:
            clabsi_df = pd.concat([clabsi_df, tdf], ignore_index=True)

        
        ################################################################
        ################## CAUTI #######################################
        ################################################################

        tdf = pd.read_pickle(mydir + "1_data/optimized_by_quarter/CAUTI/CAUTI_Data_opt_for_SIRs_" + fdate + ".pkl")
        tdf.drop_duplicates(inplace=True)
        
        tdf.rename(columns={
                            'CAUTI Predicted Cases': 'Predicted Cases',
                            'CAUTI Urinary Catheter Days': 'Days',
                            'CAUTI': 'SIR',
                            'CAUTI upper CL': 'upper CL',
                            'CAUTI lower CL': 'lower CL',
                            'CAUTI Observed Cases': 'Observed Cases',
                            }, inplace=True)
        tdf = tdf[tdf['Predicted Cases'] >= 1]
        tdf['HAI'] = 'CAUTI'
        tdf['upper CL'] = tdf['upper CL'].astype(float)
        
        days = np.array(tdf['Days'])
        pi = np.array(tdf['pi_opt'])
        z = np.array(tdf['z_opt'])
        p = (pi * days / (days + z))
        ci_low, ci_upp = proportion_confint(tdf['expected O'], days, alpha=ci_alpha, method=b_method)
        tdf['exp_random'] = days * ci_upp
        tdf['SISc'] = (tdf['Observed Cases'] - tdf['exp_random']) / (tdf['Predicted Cases'] - tdf['exp_random'])
        tdf['exp_random'] = days * p
        tdf['SISo'] = (tdf['Observed Cases'] - tdf['exp_random']) / (tdf['Predicted Cases'])
        
        tdf['exp_random'] = days * ci_upp
        tdf['SIS'] = SIS(tdf['Observed Cases'], 
                         tdf['expected O'], 
                         tdf['Predicted Cases'], 
                         tdf['upper CL'], 
                         tdf['exp_random'],
                         tdf['Days'])
        
        btr = get_btr(tdf['Observed Cases'], tdf['expected O'], tdf['exp_random'], tdf['Predicted Cases'])
        tdf['Observed is better than random'] = btr
        tdf['Winzorized SIR'] = winsorize(tdf['SIR'], limits=[0.05, 0.05])
        tdf['Winzorized SIS'] = winsorize(tdf['SIS'], limits=[0.05, 0.05])
        tdf['Winzorized SISc'] = winsorize(tdf['SISc'], limits=[0.05, 0.05])
        tdf['Winzorized SISo'] = winsorize(tdf['SISo'], limits=[0.05, 0.05])
        tdf['Winzorized z SIR'] = stats.zscore(tdf['Winzorized SIR'])
        tdf['Winzorized z SIS'] = stats.zscore(tdf['Winzorized SIS'])
        tdf['Winzorized z SISc'] = stats.zscore(tdf['Winzorized SISc'])
        tdf['Winzorized z SISo'] = stats.zscore(tdf['Winzorized SISo'])
        
        tdf['Winzorized z SISo rank'] = tdf['Winzorized z SISo'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        tdf['Winzorized z SIS rank'] = tdf['Winzorized z SIS'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        tdf['Winzorized z SISc rank'] = tdf['Winzorized z SISc'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        tdf['Winzorized z SIR rank'] = tdf['Winzorized z SIR'].rank(axis=0, method='min',
            na_option='keep', ascending=True, pct=False)
        
        if fdate == '2014-07-17':
            cauti_df = tdf.copy(deep=True)
        else:
            cauti_df = pd.concat([cauti_df, tdf], ignore_index=True)
        
        

    #########################################################################################
    ##################### WINSORIED Z-SCORE AND TOTAL HAC ###################################
    #########################################################################################

    main_df = pd.concat([mrsa_df, cdiff_df, clabsi_df, cauti_df])
    main_df.reset_index(inplace=True)
                        
    main_df['weight'] = main_df.groupby(['Facility and File Date'])['Facility and File Date'].transform("count")
    
    main_df['all better than random'] = main_df.groupby('Facility and File Date')['Observed is better than random'].transform('sum')
    main_df['all better than random'] = main_df['all better than random']  - main_df['weight']
    
    print("max(all better than random):", np.max(main_df['all better than random']))
    print("min(all better than random):", np.min(main_df['all better than random']))
    
    main_df['all better than random'].values[main_df['all better than random'] >= 0] = 1
    main_df['all better than random'].values[main_df['all better than random'] < 0] = 0
    
    main_df['weight'] = 1/main_df['weight']
    main_df['weighted Winzorized z SIS'] = main_df['Winzorized SIS'] * main_df['weight']
    main_df['weighted Winzorized z SISo'] = main_df['Winzorized SISo'] * main_df['weight']
    main_df['weighted Winzorized z SISc'] = main_df['Winzorized SISc'] * main_df['weight']
    main_df['weighted Winzorized z SIR'] = main_df['Winzorized z SIR'] * main_df['weight']

    main_df['total HAI SIS'] = main_df.groupby('Facility and File Date')['weighted Winzorized z SIS'].transform('sum')
    main_df['total HAI SISo'] = main_df.groupby('Facility and File Date')['weighted Winzorized z SISo'].transform('sum')
    
    main_df['total HAI SISc'] = main_df.groupby('Facility and File Date')['weighted Winzorized z SISc'].transform('sum')
    main_df['total HAI SIR'] = main_df.groupby('Facility and File Date')['weighted Winzorized z SIR'].transform('sum')
    
    main_df['sqrt_Days'] = np.sqrt(main_df['Days'])
    main_df['avg days'] = main_df.groupby('Facility and File Date')['sqrt_Days'].transform('mean')
    main_df.sort_values(by=['Facility and File Date', 'HAI'], inplace=True)

    print('main_df.shape:', main_df.shape)
    #print(list(main_df))
    #print(main_df.head(5))
    main_df.to_pickle(mydir + "1_data/WinsorizedZscores.pkl")

    hospitals = main_df['Facility and File Date'].str[:6]
    print(len(list(set(hospitals.tolist()))), 'hospitals')
