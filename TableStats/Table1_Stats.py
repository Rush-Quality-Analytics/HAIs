import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from math import pi
import sys
import os
import scipy as sc
import warnings

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
warnings.filterwarnings('ignore')

mydir = os.path.expanduser("~/GitHub/HAIs/")

#########################################################################################
########################## IMPORT PSI DATA ##############################################

cauti_df = pd.read_pickle(mydir + "data/CAUTI_Data.pkl")
cauti_df['provider'] = cauti_df['Facility and File Date'].str[:6]
cauti_df['date'] = cauti_df['Facility and File Date'].str[-8:]

clabsi_df = pd.read_pickle(mydir + "data/CLABSI_Data.pkl")
clabsi_df['provider'] = clabsi_df['Facility and File Date'].str[:6]
clabsi_df['date'] = clabsi_df['Facility and File Date'].str[-8:]

mrsa_df = pd.read_pickle(mydir + "data/MRSA_Data.pkl")
mrsa_df['provider'] = mrsa_df['Facility and File Date'].str[:6]
mrsa_df['date'] = mrsa_df['Facility and File Date'].str[-8:]

cdiff_df = pd.read_pickle(mydir + "data/CDIFF_Data.pkl")
cdiff_df['provider'] = cdiff_df['Facility and File Date'].str[:6]
cdiff_df['date'] = cdiff_df['Facility and File Date'].str[-8:]


print('CAUTI:', list(cauti_df), '\n')
#print(len(list(set(cauti_df['provider']))), 'hospitals')

print('CLABSI:', list(clabsi_df), '\n')
#print(len(list(set(clabsi_df['provider']))), 'hospitals')

print('MRSA:', list(mrsa_df), '\n')
#print(len(list(set(mrsa_df['provider']))), 'hospitals')

print('CDIFF:', list(cdiff_df), '\n')
#print(len(list(set(cdiff_df['provider']))), 'hospitals')

print(sorted(list(set(clabsi_df['date']))))

date = '20200129'

tdf = clabsi_df[clabsi_df['date'] == date]
ls = tdf[tdf['CLABSI'] != 'Not Available']['CLABSI'].tolist()
ls = list(np.float_(ls))
print('CLABSI:')
print('Number of hospitals:', len(ls))
print(np.nanmin(ls), np.nanmax(ls), np.nanmean(ls), np.std(ls), '\n')


tdf = cauti_df[cauti_df['date'] == date]
ls = tdf[tdf['CAUTI'] != 'Not Available']['CAUTI'].tolist()
ls = list(np.float_(ls))
print('CAUTI:')
print('Number of hospitals:', len(ls))
print(np.nanmin(ls), np.nanmax(ls), np.nanmean(ls), np.std(ls), '\n')


tdf = mrsa_df[mrsa_df['date'] == date]
ls = tdf[tdf['MRSA'] != 'Not Available']['MRSA'].tolist()
ls = list(np.float_(ls))
print('MRSA:')
print('Number of hospitals:', len(ls))
print(np.nanmin(ls), np.nanmax(ls), np.nanmean(ls), np.std(ls), '\n')


tdf = cdiff_df[cdiff_df['date'] == date]
ls = tdf[tdf['CDIFF'] != 'Not Available']['CDIFF'].tolist()
ls = list(np.float_(ls))
print('CDIFF:')
print('Number of hospitals:', len(ls))
print(np.nanmin(ls), np.nanmax(ls), np.nanmean(ls), np.std(ls), '\n')
