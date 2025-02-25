import pandas as pd
import numpy as np


def HearingClassifier(df: pd.DataFrame) -> pd.DataFrame:
    """Function that takes in audiometric frequencies and returns the degree and type of hearing loss.
    This function calculate different metrics that will be used as
    criterion for classifying the hearing loss types.

    Credits: This code was adapted from HL classification work of Michael Smith

    We pass a set of dataframes corresponding to the audiometric
    frequencies :
    'R250', 'R500', 'R1000', 'R2000', 'R3000', 'R4000', 'R6000',
    'R8000','L250', 'L500', 'L1000', 'L2000', 'L3000', 'L4000',
    'L6000', 'L8000', 'RBone500','RBone1000','RBone2000', 'RBone4000',
    'LBone500', 'LBone1000', 'LBone2000', 'LBone4000'

    Here is logic in plain English which should be easier to understand.

    Normal Hearing:
      AC < 25 dB HL
      BC < 25 dB HL
      Air-bone gap <10 dB
    Conductive:
       AC > 25 dB HL
       BC < 25 dB HL
       Air-bone gap > 10 dB
       (i.e. BC is normal but there is a hearing loss when listening via AC due to the pathology in the  outer and/or middle ear)
    Sensorineural:
       AC > 25 dB HL
       BC > 25 dB HL
       Air-bone gap <10 dB
       (i.e. there is a hearing loss present whether listening via AC or BC and there is not a significant air-bone gap)
    Mixed:
       AC > 25 dB HL
       BC > 25 dB HL
       Air-bone gap > 10 dB
       (i.e. hearing loss present but is made much worse when listening via AC because of the conductive component
        {{ e.g. BC thresholds are ~40 dB but the AC thresholds are 70 dB}})

    This code requires data with bone-conduction data to determine type of hearing loss.
    If BC threshold of one ear is missing (in case of symmetric hearing loss),then,
    BC threshold of the known ear is copied to the ear with missing BC.

    If BC threshold is missing in both ears, then the hearing loss type defaults to 'Unknown'

    Args:
      df:  dataframe with HL measurements at audiometric frequencies

    Returns:
      df: dataframe with HL classes as a new column, with several new working
      columna added.
    """

# Ensuring BC thresholds in both ears in cases of symmetric hearing loss:
# Known BC threshold from one ear copied to the other ear

    frequencies = ['500', '1000', '2000', '4000']

    right_cols = [f'RBone{freq}' for freq in frequencies]
    left_cols = [f'LBone{freq}' for freq in frequencies]

    for right_col, left_col in zip(right_cols, left_cols):
        df.fillna({right_col: left_col}, inplace=True)
        df.fillna({left_col: right_col}, inplace=True)

    # Do not calculate PTA of a ear if thresholds are 'NR' at any frequency
    NR_value = 10e5  # Label NR value as a cautionary value of a million

    # Defining required PTAs:
    pta_cols = [['R_PTA', ['R500', 'R1000', 'R2000']],
                ['L_PTA', ['L500', 'L1000', 'L2000']],
                ['R_PTA_4freq', ['R500', 'R1000', 'R2000', 'R4000']],
                ['L_PTA_4freq', ['L500', 'L1000', 'L2000', 'L4000']],
                ['R_HFPTA', ['R1000', 'R2000', 'R4000']],
                ['L_HFPTA', ['L1000', 'L2000', 'L4000']],
                ['R_LFPTA', ['R250', 'R500', 'R1000']],
                ['L_LFPTA', ['L250', 'L500', 'L1000']],
                ['R_UHFPTA', ['R2000', 'R4000', 'R8000']],
                ['L_UHFPTA', ['L2000', 'L4000', 'L8000']],
                ['R_PTA_BC', ['RBone500', 'RBone1000', 'RBone2000']],
                ['L_PTA_BC', ['LBone500', 'LBone1000', 'LBone2000']],
                ['R_HFPTA_BC', ['RBone1000', 'RBone2000', 'RBone4000']],
                ['L_HFPTA_BC', ['LBone1000', 'LBone2000', 'LBone4000']],
                ['R_PTA_BC_4freq', ['RBone500', 'RBone1000', 'RBone2000', 'RBone4000']],
                ['L_PTA_BC_4freq', ['LBone500', 'LBone1000', 'LBone2000', 'LBone4000']]]

    # Calculate PTA, but set to np.nan if any of the thresholds are NR
    for pta, cols in pta_cols:
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')  # VMA (8/19)
        df[pta] = df[cols].replace(NR_value, np.nan).mean(axis=1, skipna=False)

    # new ABGap
    df['R_PTA_ABGap'] = df['R_PTA'] - df['R_PTA_BC']
    df['R_HFPTA_ABGap'] = df['R_HFPTA'] - df['R_HFPTA_BC']
    df['R_PTA_4freq_ABGap'] = df['R_PTA_4freq'] - df['R_PTA_BC_4freq']

    df['L_PTA_ABGap'] = df['L_PTA'] - df['L_PTA_BC']
    df['L_HFPTA_ABGap'] = df['L_HFPTA'] - df['L_HFPTA_BC']
    df['L_PTA_4freq_ABGap'] = df['L_PTA_4freq'] - df['L_PTA_BC_4freq']

    # HL Type - Added 'Unknown' HL type when BC thresholds are missing in the dataframe to give 5 types of hearing loss:
    # 1. Normal
    # 2. Conductive
    # 3. SNHL
    # 4. Mixed
    # 5. Unknown

    # Right
    # 3 freq PTA using BC PTA of 500, 1k, 2k Hz
    conditions_1 = [
        (df['R_PTA_BC'] < 25.1) & (df['R_PTA_ABGap'] < 10) &
        (df['R_PTA_ABGap'] >= -20) & (df['R_PTA'] < 25),
        (df['R_PTA_BC'] < 25.1) & (df['R_PTA_ABGap'] >= 10) &
        (df['R_PTA'] > 25),
        (df['R_PTA_ABGap'] < 10) & (df['R_PTA_ABGap'] >= -20) &
        (df['R_PTA'] > 25),
        (df['R_PTA_BC'] > 25) & (df['R_PTA_ABGap'] >= 10) &
        (df['R_PTA'] > 25)
    ]
    values = ['Normal', 'Conductive', 'SNHL', 'Mixed']
    df['R_Type_HL'] = np.select(conditions_1, values, 'Unknown')

    # HFPTA of 1, 2, 4kHz
    conditions_2 = [
        (df['R_HFPTA_BC'] < 25.1) & (df['R_HFPTA_ABGap'] < 10) &
        (df['R_HFPTA_ABGap'] >= -20) & (df['R_PTA'] < 25),
        (df['R_HFPTA_BC'] < 25.1) & (df['R_HFPTA_ABGap'] >= 10) &
        (df['R_HFPTA'] > 25),
        (df['R_HFPTA_ABGap'] < 10) & (df['R_HFPTA_ABGap'] >= -20) &
        (df['R_HFPTA'] > 25),
        (df['R_HFPTA_BC'] > 25) & (df['R_HFPTA_ABGap'] >= 10) &
        (df['R_HFPTA'] > 25)
    ]

    values = ['Normal', 'Conductive', 'SNHL', 'Mixed']
    df['R_Type_HL_HF'] = np.select(conditions_2, values, 'Unknown')

    # 4 freq PTA of 500, 1k, 2k, 4kHz
    conditions_3 = [
        (df['R_PTA_BC_4freq'] < 25.1) & (df['R_PTA_4freq_ABGap'] < 10) &
        (df['R_PTA_4freq_ABGap'] >= -20) & (df['R_PTA_4freq'] < 25),
        (df['R_PTA_BC_4freq'] < 25.1) & (df['R_PTA_4freq_ABGap'] >= 10) &
        (df['R_PTA_4freq'] > 25),
        (df['R_PTA_4freq_ABGap'] < 10) & (df['R_PTA_4freq_ABGap'] >= -20) &
        (df['R_PTA_4freq'] > 25),
        (df['R_PTA_BC_4freq'] > 25) & (df['R_PTA_4freq_ABGap'] >= 10) &
        (df['R_PTA_4freq'] > 25)
    ]
    values = ['Normal', 'Conductive', 'SNHL', 'Mixed']
    df['R_Type_HL_4freq'] = np.select(conditions_3, values, 'Unknown')

    # Left
    # 3 freq PTA using the PTA of 500, 1k, 2kHz
    conditions_1 = [
        (df['L_PTA_BC'] < 25.1) & (df['L_PTA_ABGap'] < 10) &
        (df['L_PTA_ABGap'] >= -20) & (df['L_PTA'] < 25),
        (df['L_PTA_BC'] < 25.1) & (df['L_PTA_ABGap'] >= 10) &
        (df['L_PTA'] > 25),
        (df['L_PTA_ABGap'] < 10) & (df['L_PTA_ABGap'] >= -20) &
        (df['L_PTA'] > 25),
        (df['L_PTA_BC'] > 25) & (df['L_PTA_ABGap'] >= 10) &
        (df['L_PTA'] > 25)
    ]
    values = ['Normal', 'Conductive', 'SNHL', 'Mixed']
    df['L_Type_HL'] = np.select(conditions_1, values, 'Unknown')

    # HFPTA of 1k, 2k, 4kHz
    conditions_2 = [
        (df['L_HFPTA_BC'] < 25.1) & (df['L_HFPTA_ABGap'] < 10) &
        (df['L_HFPTA_ABGap'] >= -20) & (df['L_HFPTA'] < 25),
        (df['L_HFPTA_BC'] < 25.1) & (df['L_HFPTA_ABGap'] >= 10) &
        (df['L_HFPTA'] > 25),
        (df['L_HFPTA_ABGap'] < 10) & (df['L_HFPTA_ABGap'] >= -20) &
        (df['L_HFPTA'] > 25),
        (df['L_HFPTA_BC'] > 25) & (df['L_HFPTA_ABGap'] >= 10) &
        (df['L_HFPTA'] > 25)
    ]

    values = ['Normal', 'Conductive', 'SNHL', 'Mixed']
    df['L_Type_HL_HF'] = np.select(conditions_2, values, 'Unknown')

    # 4 freq PTA of 500, 1k, 2k, 4kHz
    conditions_3 = [
        (df['L_PTA_BC_4freq'] < 25.1) & (df['L_PTA_4freq_ABGap'] < 10) &
        (df['L_PTA_4freq_ABGap'] >= -20) & (df['L_PTA_4freq'] < 25),
        (df['L_PTA_BC_4freq'] < 25.1) & (df['L_PTA_4freq_ABGap'] >= 10) &
        (df['L_PTA_4freq'] > 25),
        (df['L_PTA_4freq_ABGap'] < 10) & (df['L_PTA_4freq_ABGap'] >= -20) &
        (df['L_PTA_4freq'] > 25),
        (df['L_PTA_BC_4freq'] > 25) & (df['L_PTA_4freq_ABGap'] >= 10) &
        (df['L_PTA_4freq'] > 25)
    ]
    values = ['Normal', 'Conductive', 'SNHL', 'Mixed']
    df['L_Type_HL_4freq'] = np.select(conditions_3, values, 'Unknown')

    return df
