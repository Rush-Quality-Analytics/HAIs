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

def curate(df):

    try:
        df = df[df['Facility ID'] != np.nan]
        df['Facility ID'] = df['Facility ID'].values.astype(str)
    except:
        pass
    try:
        df = df[df['Facility Name'] != np.nan]
    except:
        pass
    
    for c in list(df):    
        try:
            df[c] = df[c].str.replace("\t","")
        except:
            pass
        
    return df


main_df = pd.read_pickle(mydir + "data/Facility.pkl")
main_df.drop(['Address', 'City', 'County Name', 'Phone Number', 'ZIP Code'], 
             axis = 1, inplace=True)
main_df['file_date'] = main_df['file_date'].str[-8:]

print(list(main_df), '\n')
#print(main_df.head())
#print(list(set(main_df['file_date'])))
#sys.exit()

main_df = main_df[main_df['Measure Name'].isin([
    'Central Line Associated Bloodstream Infection: Number of Device Days',
    'CAUTI: Number of Urinary Catheter Days',
    'CAUTI Lower Confidence Limit',
    'Catheter Associated Urinary Tract Infections (ICU + select Wards): Number of Urinary Catheter Days',
    'Catheter-Associated Urinary Tract Infections (CAUTI)',
    'Catheter Associated Urinary Tract Infections (ICU + select Wards): Lower Confidence Limit',
    'CLABSI Central Line Days',
    'CAUTI: Lower Confidence Limit',
    'CLABSI: Predicted Cases',
    'Central line-associated blood stream infections (CLABSI)',
    'CLABSI Observed Cases',
    'CLABSI: Upper Confidence Limit',
    'CAUTI: Predicted Cases',
    'Central Line Associated Bloodstream Infection (ICU + select Wards): Predicted Cases',
    'CAUTI Urinary Catheter Days' ,
    'CAUTI Observed Cases',
    'Central line-associated bloodstream infections (CLABSI) in ICUs only',
    'CLABSI: Lower Confidence Limit',
    'Catheter Associated Urinary Tract Infections (ICU + select Wards)',
    'CAUTI Upper Confidence Limit',
    'CLABSI Predicted Cases',
    'CAUTI: Upper Confidence Limit',
    'Catheter-associated urinary tract infections (CAUTI) in ICUs and select wards', 
    'CLABSI Upper Confidence Limit', 
    'Central Line Associated Bloodstream Infection (ICU + select Wards)',
    'Catheter Associated Urinary Tract Infections (ICU + select Wards): Observed Cases',
    'Central Line Associated Bloodstream Infection (ICU + select Wards): Lower Confidence Limit',
    'Central Line Associated Bloodstream Infection (ICU + select Wards): Observed Cases',
    'Central line-associated bloodstream infections (CLABSI) in ICUs and select wards',
    'CLABSI Lower Confidence Limit',
    'CLABSI: Number of Procedures',
    'CLABSI: Observed Cases',
    'Central line-associated blood stream infections (CLABSI) in ICUs only',
    'Central Line Associated Bloodstream Infection (ICU + select Wards): Upper Confidence Limit', 
    'Catheter Associated Urinary Tract Infections (ICU + select Wards): Upper Confidence Limit',
    'Catheter Associated Urinary Tract Infections (ICU + select Wards): Predicted Cases',
    'CAUTI Predicted Cases',
    'Catheter-Associated Urinary Tract Infections (CAUTI) in ICUs only',
    'CAUTI: Number of Procedures',
    'CAUTI: Observed Cases',
    'CLABSI: Number of Device Days',
    ])]


main_df['Facility and File Date'] = main_df['Facility ID'] + '-' + main_df['file_date']
print(main_df['Facility and File Date'].iloc[0])

#main_df = curate(main_df)

clabsi_df = main_df[main_df['Measure Name'].isin([
    'Central Line Associated Bloodstream Infection: Number of Device Days',
    'CLABSI Central Line Days',
    'CLABSI: Predicted Cases',
    'Central line-associated blood stream infections (CLABSI)',
    'CLABSI Observed Cases',
    'CLABSI: Upper Confidence Limit',
    'Central Line Associated Bloodstream Infection (ICU + select Wards): Predicted Cases',
    'Central line-associated bloodstream infections (CLABSI) in ICUs only',
    'CLABSI: Lower Confidence Limit',
    'CLABSI Predicted Cases',
    'CLABSI Upper Confidence Limit', 
    'Central Line Associated Bloodstream Infection (ICU + select Wards)',
    'Central Line Associated Bloodstream Infection (ICU + select Wards): Lower Confidence Limit',
    'Central Line Associated Bloodstream Infection (ICU + select Wards): Observed Cases',
    'Central line-associated bloodstream infections (CLABSI) in ICUs and select wards',
    'CLABSI Lower Confidence Limit',
    'CLABSI: Number of Procedures',
    'CLABSI: Observed Cases',
    'Central line-associated blood stream infections (CLABSI) in ICUs only',
    'Central Line Associated Bloodstream Infection (ICU + select Wards): Upper Confidence Limit', 
    'CLABSI: Number of Device Days',
    ])]


#f = main_df['Facility and File Date'].iloc[-1]
#tdf = main_df[main_df['Facility and File Date'] == f]
#print(list(set(tdf['Measure Name'].tolist())))
#sys.exit()


d = {'Central Line Associated Bloodstream Infection: Number of Device Days': 'CLABSI Number of Device Days',
     'CLABSI: Number of Device Days': 'CLABSI Number of Device Days',
     'CLABSI Central Line Days': 'CLABSI Number of Device Days',
                                                         
     'CLABSI Lower Confidence Limit': 'CLABSI lower CL',
     'CLABSI: Lower Confidence Limit': 'CLABSI lower CL',
     'Central Line Associated Bloodstream Infection (ICU + select Wards): Lower Confidence Limit': 'CLABSI lower CL',
                                                         
     'CLABSI Upper Confidence Limit': 'CLABSI upper CL',
     'CLABSI: Upper Confidence Limit': 'CLABSI upper CL',
     'Central Line Associated Bloodstream Infection (ICU + select Wards): Upper Confidence Limit': 'CLABSI upper CL',
          

                                               
     'Central line-associated blood stream infections (CLABSI) in ICUs only': 'CLABSI',
     'Central Line Associated Bloodstream Infection (ICU + select Wards)': 'CLABSI',
     'Central line-associated bloodstream infections (CLABSI) in ICUs and select wards': 'CLABSI',
     'Central line-associated blood stream infections (CLABSI)': 'CLABSI',
     'Central line-associated bloodstream infections (CLABSI) in ICUs only': 'CLABSI',
                                                    
     'CLABSI: Observed Cases': 'CLABSI Observed Cases',
     'Central Line Associated Bloodstream Infection (ICU + select Wards): Observed Cases': 'CLABSI Observed Cases',
                                                         
     'Central Line Associated Bloodstream Infection (ICU + select Wards): Predicted Cases': 'CLABSI Predicted Cases',
     'CLABSI: Predicted Cases': 'CLABSI Predicted Cases',
                                                         
     'CLABSI: Observed Cases': 'CLABSI Observed Cases',
     'Central Line Associated Bloodstream Infection (ICU + select Wards): Observed Cases': 'CLABSI Observed Cases',
                                                         
     'CLABSI: Number of Procedures': 'CLABSI Number of Procedures',
     }

clabsi_df['Measure Name'].replace(to_replace=d, inplace=True)

print(list(set(clabsi_df['Measure Name'].tolist())))
#sys.exit()

df = pd.DataFrame(columns=['Facility and File Date'])
IDs = list(set(clabsi_df['Facility and File Date'].tolist()))

dates = []
CLABSI = [] 
IDs2 = []
IDs3 = []
measures = ['CLABSI', 'CLABSI Predicted Cases', 'CLABSI Number of Device Days', 
            'CLABSI upper CL', 'CLABSI Number of Procedures', 'CLABSI Observed Cases', 
            'CLABSI lower CL']

measure_lists = [ [] for _ in range(len(measures)) ]

for i, ID in enumerate(IDs):
    print(len(IDs) - i)
    
    #if i > 10: break
    
    tdf = clabsi_df[clabsi_df['Facility and File Date'] == ID]
    
    dates.append(tdf['file_date'].iloc[0])
    IDs2.append(ID)
    IDs3.append(tdf['Facility ID'].iloc[0])
    
    for j, measure in enumerate(measures):
        tdf2 = 0
        try:
            tdf2 = tdf[tdf['Measure Name'] == measure]
            val = tdf2['Score'].iloc[0]
        except:
            val = np.nan
            
        measure_lists[j].append(val)
        
df['Facility and File Date'] = IDs2
df['Facility ID'] = IDs3

for i, measure in enumerate(measures):
    df[measure] = measure_lists[i]
df['file date'] = dates

print(list(df), '\n')

df['file date'] = pd.to_datetime(df['file date'], format="%Y%m%d")

df.to_pickle(mydir + "data/CLABSI_Data.pkl")
df.to_csv(mydir + "data/CLABSI_Data.csv")
