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
#print(list(set(main_df['Measure Name'])))
#sys.exit()

main_df = main_df[main_df['Measure Name'].isin([
    'Clostridium Difficile (C.Diff): Observed Cases', 
    'Clostridium Difficile (C.Diff)', 
    'C.diff Observed Cases', 
    'Clostridium Difficile (C.Diff): Patient Days', 
    'Clostridium difficile (C.diff.) Laboratory-identified Events (Intestinal infections)', 
    'Clostridium difficile (C.diff.) intestinal infections', 
    'C.diff Upper Confidence Limit', 
    'Clostridium Difficile (C.Diff): Lower Confidence Limit', 
    'C.diff Predicted Cases', 
    'CLABSI: Upper Confidence Limit', 
    'Clostridium Difficile (C.Diff): Upper Confidence Limit', 
    'Clostridium Difficile (C.Diff): Predicted Cases', 
    'C.diff Lower Confidence Limit', 
    'C.diff Patient Days',
    
    'MRSA Lower Confidence Limit', 
    'MRSA Bacteremia: Lower Confidence Limit', 
    'Methicillin-resistant Staphylococcus Aureus (MRSA) Blood Laboratory-identified Events (Bloodstream infections)', 
    'Methicillin-resistant Staphylococcus Aureus (MRSA) blood infections', 
    'MRSA Observed Cases', 
    'MRSA Predicted Cases', 
    'MRSA Bacteremia', 
    'MRSA Upper Confidence Limit', 
    'MRSA Bacteremia: Patient Days', 
    'MRSA Bacteremia: Observed Cases', 
    'MRSA Patient Days', 
    'MRSA Bacteremia: Upper Confidence Limit', 
    'MRSA Bacteremia: Predicted Cases', 
    
    ])]


main_df['Facility and File Date'] = main_df['Facility ID'] + '-' + main_df['file_date']
print(main_df['Facility and File Date'].iloc[0])

#main_df = curate(main_df)

mrsa_df = main_df[main_df['Measure Name'].isin([
    'MRSA Lower Confidence Limit', 
    'MRSA Bacteremia: Lower Confidence Limit', 
    
    'Methicillin-resistant Staphylococcus Aureus (MRSA) Blood Laboratory-identified Events (Bloodstream infections)', 
    'Methicillin-resistant Staphylococcus Aureus (MRSA) blood infections',
    'MRSA Bacteremia',
    
    'MRSA Observed Cases',
    'MRSA Bacteremia: Observed Cases',
    
    'MRSA Predicted Cases',
    'MRSA Bacteremia: Predicted Cases',

    'MRSA Upper Confidence Limit',
    'MRSA Bacteremia: Upper Confidence Limit',
    
    'MRSA Bacteremia: Patient Days',
    'MRSA Patient Days',
    
    ])]


d = {
     'MRSA Lower Confidence Limit': 'MRSA lower CL',
     'MRSA Bacteremia: Lower Confidence Limit': 'MRSA lower CL',
    
     'Methicillin-resistant Staphylococcus Aureus (MRSA) Blood Laboratory-identified Events (Bloodstream infections)': 'MRSA',
     'Methicillin-resistant Staphylococcus Aureus (MRSA) blood infections': 'MRSA',
     'MRSA Bacteremia': 'MRSA',
    
     'MRSA Observed Cases': 'MRSA Observed Cases', 
     'MRSA Bacteremia: Observed Cases': 'MRSA Observed Cases',
    
     'MRSA Predicted Cases': 'MRSA Predicted Cases', 
     'MRSA Bacteremia: Predicted Cases': 'MRSA Predicted Cases',

     'MRSA Upper Confidence Limit': 'MRSA upper CL', 
     'MRSA Bacteremia: Upper Confidence Limit': 'MRSA upper CL',
    
     'MRSA Bacteremia: Patient Days': 'MRSA patient days',
     'MRSA Patient Days': 'MRSA patient days',
     
     }

mrsa_df['Measure Name'].replace(to_replace=d, inplace=True)

print(list(set(mrsa_df['Measure Name'].tolist())))
#sys.exit()

df = pd.DataFrame(columns=['Facility and File Date'])
IDs = list(set(mrsa_df['Facility and File Date'].tolist()))

dates = []
MRSA = [] 
IDs2 = []
IDs3 = []
measures = ['MRSA Predicted Cases', 'MRSA patient days', 'MRSA', 'MRSA upper CL', 
            'MRSA lower CL', 'MRSA Observed Cases']

measure_lists = [ [] for _ in range(len(measures)) ]

for i, ID in enumerate(IDs):
    print(len(IDs) - i)
    
    tdf = mrsa_df[mrsa_df['Facility and File Date'] == ID]
    
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

df.to_pickle(mydir + "data/MRSA_Data.pkl")
df.to_csv(mydir + "data/MRSA_Data.csv")
