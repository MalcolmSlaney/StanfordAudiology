# -*- coding: utf-8 -*-
"""
Created on Thurs May 23 14:48:24 2024

@author: varsh
"""

import pandas as pd

def extract_specific_icd (input_csv, output_csv, icd9_codes=None, icd10_codes=None):
    df = pd.read_csv(input_csv, on_bad_lines='warn')
    
    df = df.astype(str)  # Columns read as strings to handle any non-numeric issues
    
    # Initialize empty lists if no specific codes are provided
    if icd9_codes is None:
        icd9_codes = []
    if icd10_codes is None:
        icd10_codes = []
   
    #Match ICD-10 codes only up to the first decimal - To not specify all associated codes with a specific disorder 
    def match_icd_code(code, code_list):
        return any(code.startswith(c) for c in code_list)

    icd9 = df['ICD9 Code'].apply(lambda x: match_icd_code(x, icd9_codes))
    icd10 = df['ICD10 Code'].apply(lambda x: match_icd_code(x, icd10_codes))
     
    combined_icd = icd9 | icd10
    
    new_df = df[combined_icd]
    
    # # Sort by Patient Id and Date to ensure the latest visit is identified - Not working, check 
    # new_df['Date'] = pd.to_datetime(new_df['Date'])
    # new_df = new_df.sort_values(by=['Patient Id', 'Date'])

    # # Drop duplicate patient IDs, keeping only the last entry (latest visit) for each patient
    # final_df = new_df.drop_duplicates(subset='Patient Id', keep='last')
    
    output_df = new_df[['Patient Id', 'Date', 'Age', 'ICD9 Code', 'ICD10 Code', 'Description']]
    
    # Save the filtered data to a new CSV file
    output_df.to_csv(output_csv, index=False)


input_csv = 'C:/Users/varsh/Desktop/PostDoc/Audiology_Data/Raw/MD_H81.0/MD_Diagnosis_H81.0.csv'
output_csv = 'C:/Users/varsh/Desktop/PostDoc/Audiology_Data/Processed/MD_H81.0/MD_Refined_RemoveMultipleVisits_5.24.csv' #-- Not working

# Meniere's here 
icd9_codes = ['386']
icd10_codes = ['H81.0']

extract_specific_icd (input_csv, output_csv, icd9_codes=icd9_codes, icd10_codes=icd10_codes)

