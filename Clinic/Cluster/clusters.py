"""Code to read in the Stanford Audiology dataset and compute the clusters.
This code also computes the HL types, and the SII before saving a copy of the
data in Google Drive.
"""
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from typing import Dict, List, Optional, Tuple, Union
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import json
# from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
import os
import datetime

import gspread
import google.auth as auth  # authenticating to google

#Global variables/parameters

default_cluster_dir = ('/content/drive/MyDrive/Stanford Audiology Models/'
                       'Colab Notebooks/')

"""
A change is required in the spreadsheet_path. Open the spreadsheet mfb raw audio
and copy the link from the address bar. The gid code in the url is unique for
each of the user.
"""
spreadsheet_path_v1 = ('https://docs.google.com/spreadsheets/d/'
                       '119_-qrfzGXwV1YBUJdBzvtAQTZnl-xwN7hD9FK5SWfU/'
                       'edit#gid=84023254')

duplicate_column_name_v1: str = 'LBone2000'

#Golden cluster can be computed by calling CreateKMeans() with random_seed = 0

golden_cluster_v1 = {(69.764465, 17.85707): 'High flat',
                     (22.426544, 19.861082): 'Low slope',
                     (42.734375, 5.0265923): 'Mid flat',
                     (35.734127, 46.22456): 'Mid slope',
                     (9.56347, 2.5132303): 'Low flat',
                     (52.65366, 47.10048): 'High slope'}

cluster_labels_v1 = {4: 'Low flat',
                     2: 'Mid flat',
                     0: 'High flat',
                     1: 'Low slope',
                     3: 'Mid slope',
                     5: 'High slope'}
labels_v1 = ['R250', 'R500', 'R1000', 'R2000',
             'R3000', 'R4000', 'R6000', 'R8000']

######################  DATA IMPORT  ############################


def ImportSpreadsheet(path) -> List:
  """
  Function that imports the spreadsheet from the specified `path`

  Before calling this function do this:
    import google.colab as colab
    colab.auth.authenticate_user()

  Args:
    - path - Filepath to the spreadsheet [GCloud URL]
  Return:
    - rows - Each row of the spreadsheet is stored as a list. First row of the
    - list contains feature names.
  """
  creds, _ = auth.default()
  gc = gspread.authorize(creds) # get_all_values gives a list of rows
  worksheet = gc.open_by_url(path).sheet1
  rows = worksheet.get_all_values()

  return rows


def ReadData(duplicate_column_name: str = duplicate_column_name_v1,spreadsheet_path: str = spreadsheet_path_v1):
  """
  Clean and transform data from a spreadsheet into a numpy array.

  Parameters:
    features (List[str], optional):
    A list of strings representing the names of the features to include
    in the DataFrame. Defaults to the value of `features_before`.
    duplicate_column_name (str, optional):
    The name of the duplicate column to handle in the data.
    Defaults to the value of `duplicate_column_name_1`.
    spreadsheet_path (str, optional): The file path of the spreadsheet
    containing the data to process. Defaults to the value of
    `spreadsheet_v1_path`.

  Returns:
    data: A list of lists.  The first list has the column names.  Succeeding
    lists contain the actual data
  """

  rows = ImportSpreadsheet(spreadsheet_path)
  rows = RenameDuplicateColumns(rows, duplicate_column_name)
  return rows


def MakePandas(rows_of_data: List) -> pd.DataFrame:
  """
  Creates a Pandas DataFrame from a list of rows containing data.

  Args:
    rows_of_data (List): A list of rows, where each row is an iterable containing data values.
    Rows of data obtained after calling
    RenameDuplicateColumns() on `rows_of_data`
    Column_names are first list element of `rows_of_data`
  Returns:
    pd.DataFrame: A Pandas DataFrame containing the converted data.
  """
  features = ConvertSpaces(rows_of_data[0])
  data = ConvertToNumerical(rows_of_data[1:])
  data_df = pd.DataFrame(data, columns = features)

  return data_df


def ImportHearingSpreadsheet(
    spreadsheet_path: str = spreadsheet_path_v1,
    duplicate_column_name: str = duplicate_column_name_v1,
    labels: Union[List[str], str] = 'default_v1') -> pd.DataFrame:
  """Read and clean Stanford hearing data from spreadsheet.

  Args:
    spreadsheet_path: Where to find the spreadsheet with the golden data
    duplicate_column_names: Which columns are duplicated in the spreadsheet and should be removed
    labels: Which column names should be used for clustering

  Returns:
    A panda dataframe with all the data.
  """
  if labels == 'default_v1':
    labels = labels_v1

  # Read data and create DataFrame
  data = ReadData(duplicate_column_name, spreadsheet_path)
  df = MakePandas(data)

  # Remove rows with invalid ages
  df_good_age = RemoveRowsWithBadAges(df)

  # Apply HLossClassifier and further data processing
  df = HLossClassifier(df_good_age)
  df = RemoveRowsWithBCWorseAC(df)

  # Prepare data for clustering
  hl_data = df.dropna(subset=labels)
  # hl_data = hl_data[labels]

  return hl_data

######################  HL CLASSIFICATION  ############################


def HLossClassifier(df: pd.DataFrame) -> pd.DataFrame:
  """
  Function that takes in audiometric frequencies and returns the hearling
  loss class. This function calculate different metrics that will be used as
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
  df['L_Type_HL_HF'] = np.select(conditions_2, values, 'Unkwown')

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


def HLPlot(df: pd.DataFrame, title = None):
  """
  This function displays a scatter plot of Avg air conduction vs Avg bone
  conduction threshold coloured by 4 hearing loss classes:
    - Normal
    - SNHL
    - Conductive
    - Mixed
  Args:
    - df: dataframe computed by function HL_loss_classification()
    - title: title of the plot (incase when needed)

  Displays: a scatter plot in the current figure.

  """
  df_normal = df.loc[df['R_Type_HL_All'] == 'Normal']
  df_snhl = df.loc[df['R_Type_HL_All'] == 'SNHL']
  df_conductive = df.loc[df['R_Type_HL_All'] == 'Conductive']
  df_mixed = df.loc[df['R_Type_HL_All'] == 'Mixed']

  plt.scatter(df_normal['R_PTA_BC_All'], df_normal['R_PTA_All'], color='r', alpha = 0.5, label = 'Normal')
  plt.scatter(df_snhl['R_PTA_BC_All'], df_snhl['R_PTA_All'], color='b', alpha = 0.5, label = 'SNHL')
  plt.scatter(df_conductive['R_PTA_BC_All'], df_conductive['R_PTA_All'], alpha = 0.5, color = 'g', label = 'Conductive')
  plt.scatter(df_mixed['R_PTA_BC_All'], df_mixed['R_PTA_All'], color = 'black', alpha = 0.5, label = 'Mixed')

  plt.xlabel('Average Bone Conduction Threshold')
  plt.ylabel('Average Air Conduction Threshold')
  plt.title(title)
  plt.xlim(-20, 100)
  plt.ylim(-20, 130)
  plt.legend()
  plt.show()

######################  DATA CLEANSING  ############################


def RenameDuplicateColumns(row_list: List, column_name: str) -> List:
  """
  Function to rename a duplicate column name.

  Args:
    - row_list: A list containing each row_list of the spreadsheet
    - column_name: Name of the duplicate column

  Returns:
   - A list containing no duplicate for the specified column_name
  """
  indices = [i for i in range(len(row_list)) if row_list[i] == column_name]
  if len(indices) > 1:
    row_list[indices[1]] = column_name+'Dupe'

  return row_list


def ConvertSpaces(column_names: List) -> List:
  """
  A function to convert spaces in all column names to _
  so they can be field names.

  Args:
    - column_names: A list containing column names of the spreadsheet

  Returns:
    - A list containing no _ in columns names
  """
  column_names = [s.replace(' ', '_') for s in column_names]
  return column_names


def ConvertToNumerical(rows_of_data: List, desired_type=np.float32) -> np.ndarray:
  """
  Converts a list of rows containing data into a numerical NumPy array.
  If a value is missing --> NaN
  If hearing threshold is NR --> 1000000 - User can modify if absolutely necessary

  Args:
    rows_of_data (List): A list of rows, where each row is an iterable containing data values.
    desired_type (data type, optional): The data type to use for the NumPy array. Default is np.float32.

  Returns:
    np.ndarray: A numerical NumPy array containing the converted data.
  """

  data = np.zeros((len(rows_of_data), len(rows_of_data[0])), dtype=desired_type)

  for i, r in enumerate(rows_of_data):
    for j, d in enumerate(r):

      if desired_type == np.float32 or type == np.float64:
        if d == '':
          d = np.nan
        elif d == 'NR':
          d == 1000000
        else:
          try:
            d = float(d)
          except ValueError:   # Last resort
            d = np.nan

      data[i, j] = d

  return data


def RemoveRowsWithBadAges(df: pd.DataFrame) -> pd.DataFrame:
  """
  Removes rows from a DataFrame with invalid age values.

  Args:
    df (pd.DataFrame): The input DataFrame containing age data.

  Returns:
    pd.DataFrame: A new DataFrame with rows removed if their 'AgeAtTestDate' values are outside the valid range [0, 100].
  """

  df_good_age = df.drop(df.loc[(df['AgeAtTestDate'] < 0) | (df['AgeAtTestDate'] > 100)].index)
  return df_good_age


def RemoveRowsWithBCWorseAC(data: pd.DataFrame, threshold: int = 100) -> pd.DataFrame:
  """
  A function to drop the rows where Bone Conduction is greater than Air
  Conduction by at least 100 dB

  Args:
  - data: A pandas dataframe on which HLossClassifier() function is run.
  - threshold: The minimum difference between air conduction and bone conduction loss.
  Return:
  - data: A pandas dataframe where the rows with bone conduction loss > air conduction loss are dropped
  """

  initial_row_count = data.shape[0]

  data = data.drop(data.loc[data['R_PTA_BC_4freq'] - data['R_PTA_4freq'] >=
                threshold].index)
  data = data.drop(data.loc[data['L_PTA_BC_4freq'] - data['L_PTA_4freq'] >=
                threshold].index)

  final_row_count = data.shape[0]

  print('Number of rows that have been dropped due to AC/BC differences: ',
      initial_row_count - final_row_count)

  return data

##################  CLUSTERING via KMeans  ############################


def CreateKMeans(n: int,
                 data: pd.DataFrame,
                 random_state: int = 0,
                 max_iter: int = 1000,
                 n_init: int = 10) -> sklearn.cluster._kmeans.KMeans:
  if n <= len(data):
    kmeans = KMeans(n_clusters=n,
                    init='k-means++',
                    max_iter=max_iter,
                    n_init=n_init,
                    random_state=random_state)
    kmeans.fit(data)  # To compute cluster centres
    return kmeans


def PlotClusterCenters(
    labels: List[str],
    kmeans: sklearn.cluster._kmeans.KMeans,
    cluster_label: Optional[Union[dict, str]] = 'default_v1',
    n: int = 6):
  """
  This function plots the clusters created using create_k_means

  Labels are of the form string(XNNN) where X is L or R for left and right ears,
  and NNN is the frequemcy. Hence, we extract only int(NNN) from the labels

  Args:
    - labels: Labels for x-axis
    - kmeans: kmeans clusters computed using scikit learn package
    - cluster_label: A dictionary that assigns predefined labels to clusters (default value is cluster_labels_v1)
    - n: Number of clusters (default value is 6)
  """
  if cluster_label == 'default_v1':
    cluster_label = cluster_labels_v1

  # color = plt.cm.rainbow(np.linspace(0, 1, 6))

  conv_labels = [int(value[1:]) for value in labels]

  # Check the shape of the cluster centers array
  if kmeans.cluster_centers_.ndim == 0:
    raise ValueError('The cluster centers array must be at least 1-dimensional')

  if cluster_label is None:
    cluster_label = [f'Cluster {i}' for i in range(n)]
  # Plot
  for i in range(n):
    plt.semilogx(conv_labels, kmeans.cluster_centers_.T[:, i], label = cluster_label[i])
    plt.legend()

  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Mean Hearing Loss (dB)')
  plt.title(f'{n} Clusters formed with respect to the standard frequencies')
  plt.show()


def TestKMeansClusters():
  """
  Tests the clustering algorithm by generating n clusters of data points,
  and then clustering them using the KMeans algorithm. The function then
  verifies that the cluster centers are within a certain threshold of 0,
  and that the clusters contain points that are within a certain range of
  x values.

  """

  np.random.seed(42)  # For reproducibility

  # Parameters
  variance = 0.5  # Variance of the Gaussian blobs
  num_points = 1000  # Total number of points
  n = 4  # Number of clusters

  # Generate data using make_blobs
  centers = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])  # Centers of the blobs
  data, _ = make_blobs(n_samples=num_points, centers=centers, cluster_std=variance)

  # Populate DataFrame with the data
  df = pd.DataFrame(data, columns=['X', 'Y'])

  # Perform clustering using KMeans
  kmeans = CreateKMeans(n, data)
  cluster_labels = kmeans.fit_predict(df[['X', 'Y']])
  df['Cluster'] = cluster_labels

  # Verify cluster conditions
  valid_centers = []
  valid_clusters = []
  for i in range(n):
    cluster_data = df[df['Cluster'] == i]
    mean_y = cluster_data['Y'].mean()
    print(np.abs(mean_y))
    valid_centers.append(np.abs(mean_y) < 0.01)
    # checking if the cluster number is in X itself
    valid_clusters.append(i in cluster_data['X'].unique())

  # Print the verification results
  print('Cluster Centers Verification (|y| < 0.1):', any(valid_centers))
  print('Clusters Verification (x in {0, 1, 2, 3}):',
      list(sorted(df['Cluster'].unique())) == [0, 1, 2, 3])


def KMeansPredictions(kmeans: sklearn.cluster._kmeans.KMeans,
                      data: pd.DataFrame,
                      # new_column_name: str = 'prediction'
                      ) -> pd.DataFrame:
  """
  Predicts the cluster labels for the given data using the given KMeans model.

  Args:
    - kmeans: The KMeans model to be used for predicting the cluster labels.
    - data: The data to be clustered
    - new_column_name: The name of the new column containing the cluster assignments.

  Returns:
    A new data frame enhanced with a new column containing the cluster assignment.
  """
  p = kmeans.predict(data)
  return p


def CountPredictions(data: pd.DataFrame,
                     # cluster_label: Optional[dict] = None,
                     cluster_field='predictions') -> Dict[int, int]:
  """
  Counts the number of patients in each cluster.

  Args:
    data: The DataFrame containing the HL classified data.
    cluster_label: An optional dictionary that maps cluster labels to
    their names.

  Returns:
    A dictionary that maps cluster labels to the number of patients in
    each cluster.
  """
  return Counter(data[cluster_field])


def AssignClusterLabels(
    data: pd.DataFrame,
    cluster_label: Union[dict, str] = 'default_v1') -> pd.DataFrame:
  """Assigns cluster labels to the data.

  Args:
    data: The data to assign cluster labels to.
    cluster_label: A dictionary mapping cluster IDs to cluster labels.

  Returns:
    The data with the cluster labels assigned.
  """
  if cluster_label == 'default_v1':
    cluster_label = cluster_labels_v1

  data['cluster_labels'] = float('nan')
  for i in list(cluster_label.keys()):
    data.loc[data['predictions'] == i, 'cluster_labels'] = cluster_label[i]
  return data


def ComputeCentroids(
    data: pd.DataFrame,
    num_clusters: int,
    label_names: List[str] = labels_v1):
  """Go through the pre-computed clusters and recompute the centroid centers.
  Use the cluster label for each patient from the dataframe.
  Also compute the standard deviations of each cluster.
  This is also a way to make sure the data is consistent.
  Args:
    data: The dataframe containing all the HL data
    num_clusters: Which precomputed cluster count to analyze
    label_names: The column names (freqs) used to compute the clusters

  """
  cluster_column = f'Cluster{num_clusters:02d}Way'
  centroids = np.zeros((num_clusters, len(label_names)))
  stds = np.zeros((num_clusters, len(label_names)))
  for i in range(num_clusters):
    these_rows = data[data[cluster_column] == i]
    these_hls = these_rows[label_names]
    centroids[i, :] = np.mean(these_hls, axis=0)
    stds[i, :] = np.std(these_hls, axis=0)
  return centroids, stds


def ComputeDistances(
    data: pd.DataFrame,
    num_clusters: int,
    label_names: List[str] = labels_v1) -> pd.DataFrame:
  """Compute the distances from the winning centroid to each datapoint.  Return
  a new panda dataframe (with the same key).   Use the cluster label for each
  patient from the dataframe.

  The distances are scaled by the dimensionality of the centroid, so the
  distances are HL/frequency and are in dB.
  """
  cluster_column = f'Cluster{num_clusters:02d}Way'
  results = []
  for i in range(num_clusters):
    these_rows = data[data[cluster_column] == i]
    these_hls = these_rows[label_names]
    # print(these_hls.shape, type(these_hls))
    centroids = np.mean(these_hls, axis=0)
    these_distances = np.sqrt(np.sum((these_hls - centroids)**2, axis=1)/len(label_names))
    results.append(pd.DataFrame(these_distances, index=these_rows.index))
  results = pd.concat(results)
  return results

#####################  DATA SAVING UTILITIES  ############################


def SaveAsJson(kmeans: sklearn.cluster._kmeans.KMeans,
               features_initial: List,
               features_final: List,
               num_cluster_points: dict,
               filename,
               path: str = default_cluster_dir,
               cluster_labels: Union[dict, str] = 'default_v1') -> str:
  """
  Save the KMeans clustering results as a JSON file.

  Args:
    - path: The path of the google directory where the JSON file will be saved.
    - cluster_labels: A dictionary mapping cluster labels to the corresponding data points (default is cluster_labels_v1)
    - kmeans: The KMeans model that was used to cluster the data.
    - features_initial: The list of features that were used to cluster the data.
    - features_final: Additional features computed when performing the HL classifcation.
    - filename: Name of the file
    - num_cluster_points: A dictionary mapping cluster labels to the number of data points in each cluster
    - clusters: A dictionary that maps the mean and slope of cluster centroids to the labels (default is golden_cluster_v1)

  Returns:
    The path to the JSON file that was saved.
  """
  if cluster_labels == 'default_v1':
    cluster_labels = cluster_labels_v1

  data = {'cluster_centers': kmeans.cluster_centers_.tolist(),
          'random_state': kmeans.random_state,
          'n_init': kmeans.n_init,
          'n_cluster': kmeans.n_clusters,
          'max_iter': kmeans.max_iter,
          'features_before': features_initial,
          'features_after': features_final,
          'cluster_labels': cluster_labels,
          'n_cluster_points': num_cluster_points,
          'date_time': str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
          }
  filepath = os.path.join(path + filename + '.json')
  with open(filepath, 'w', encoding='utf8') as file:
    json.dump(data, file, indent=4)

  return filepath


def ChangeKeyValuesToInteger(dictionary):
  """
  Change the key values of a dictionary from string to integer.

  This function converts the string keys of the cluster_labels dictionary,
  which are retrieved from the cluster_info_*.json file.

  Args:
    dictionary: The dictionary whose key values will be changed.

  Returns:
    A new dictionary with the key values changed to integer.
  """

  new_dictionary = {}
  for key, value in dictionary.items():
    key_as_integer = int(key)
    new_dictionary[key_as_integer] = value

  return new_dictionary


def LoadFromJson(path: str = default_cluster_dir,) -> Tuple[KMeans, List[str], List[str], List[str]]:
  """
  Load the KMeans clustering results from a JSON file.

  Args:
    path: The path of the google directory where the JSON file will be loaded
    from.

  Returns:
    A tuple of the KMeans model, features before HL classification,
    cluster labels and the list of features after HL classification.
  """

  with open(path, 'r', encoding='utf8') as file:
    info = json.load(file)

  cluster_centers = np.array(info['cluster_centers'])
  random_state = info['random_state']
  n_init = info['n_init']
  n_cluster = info['n_cluster']
  max_iter = info['max_iter']
  features_before = info['features_before']
  features_after = info['features_after']
  # date_time = info['date_time']
  # clusters_dict = info['clusters']  # What is this for?

  # Initialise a kmeans object
  kmeans = KMeans()
  kmeans.cluster_centers_ = cluster_centers.astype(np.double)
  kmeans.random_state = random_state
  kmeans.n_init = n_init
  kmeans.n_clusters = n_cluster
  kmeans.max_iter = max_iter
  kmeans._n_threads = 1  # _openmp_effective_n_threads()

  return kmeans, features_before, features_after

#######################  CLUSTER PROCESSING  ############################


def SlopeandMean(kmeans: sklearn.cluster._kmeans.KMeans):
  """
  Calculate the slopes and means of the cluster centers obtained from a k-means
  clustering algorithm.

  Parameters:
  kmeans : sklearn.cluster._kmeans.KMeans
    The result of a k-means clustering algorithm containing cluster centers.

  Returns:
  list of tuples
    A list of tuples, where each tuple contains the mean and slope
    corresponding to a cluster center.
  """

  slopes = kmeans.cluster_centers_.T[7, :] - kmeans.cluster_centers_.T[0, :]
  means = np.mean(kmeans.cluster_centers_, axis=1)
  slope_mean = list(zip(means, slopes))
  return slope_mean


def EuclideanDistance(centroid1, centroid2) -> float:
  """
    Calculate the Euclidean distance between two cluster centroids represented
    as mean and slope.

    Parameters:
    centroid1 : The first centroid
    centroid2 : The second centroid

    Returns:
    float: The Euclidean distance between the two centroids.

  """
  return np.linalg.norm(np.array(centroid1) - np.array(centroid2))


def CreateClusterLabels(
    kmeans: sklearn.cluster._kmeans.KMeans,
    ref_cluster: Union[dict, str] = 'default_v1') -> Dict[int, str]:
  """
    Assign cluster labels to centroids based on their proximity to the centroids
    in the `golden_cluster` dictionary.

    Parameters:
    kmeans : object
    The result of a k-means clustering algorithm containing centroids and
    cluster assignments.

    ref_cluster : dict, optional
    A dictionary representing the `golden_cluster` centroids. It should
    have centroids as keys and cluster labels as values.
    The default value is `golden_cluster`, which may represent some
    pre-defined "golden" cluster centroids.

    Returns:
      A dictionary containing the updated cluster labels for each centroid in `kmeans`.
      The keys represent the index of the centroid in `kmeans`, and the values are the corresponding cluster labels from the `golden_cluster` dictionary.
  """
  if ref_cluster == 'default_v1':
    ref_cluster = golden_cluster_v1

  clusters_slope_mean = SlopeandMean(kmeans)
  new_labels = {}
  golden_cluster_centroids = list(ref_cluster.keys())

  for i, (mean, slope) in enumerate(clusters_slope_mean):
    distance_mean = [EuclideanDistance(mean, centroid[0])
                     for centroid in golden_cluster_centroids]
    distance_slope = [EuclideanDistance(slope, centroid[1])
                      for centroid in golden_cluster_centroids]
    index_mean = np.argmin(distance_mean)
    index_slope = np.argmin(distance_slope)

    if index_mean == index_slope:
      index = index_mean
      new_labels[i] = ref_cluster[golden_cluster_centroids[index]]

  return new_labels


def CreateClusterV1(filename: str,
                    duplicate_column_name: str = duplicate_column_name_v1,
                    cluster_features: Union[List[str], str] = 'default_v1',
                    spreadsheet_path: Union[str, pd.DataFrame] = spreadsheet_path_v1,
                    save_path: str = default_cluster_dir,
                    # ref_cluster: dict = golden_cluster_v1,
                    random_state: int = 0,
                    n: int = 6,
                    max_iter: int = 1000,
                    n_init: int = 10):
  if cluster_features == 'default_v1':
    cluster_features = labels_v1

  if isinstance(spreadsheet_path, str):
    all_data = ImportHearingSpreadsheet(spreadsheet_path,
                                        duplicate_column_name,
                                        cluster_features)
  elif isinstance(spreadsheet_path, pd.DataFrame):
    all_data = spreadsheet_path
  else:
    raise TypeError('Argument must be a str or DataFrame')
  print(f'all_data has {len(all_data.columns)} columns')
  # Create and apply K-means clustering
  kmeans = CreateKMeans(n, all_data[cluster_features], random_state, max_iter, n_init)
  # cluster_labels = CreateClusterLabels(kmeans, ref_cluster)
  prediction_col_name = 'predictions'
  # Do KMeans clustering over a subset of the original data
  cluster_ids = KMeansPredictions(kmeans, all_data[cluster_features])
  assert len(all_data) == len(cluster_ids)
  all_data[prediction_col_name] = cluster_ids  # hl_data[prediction_col_name]

  # Assign cluster labels and calculate cluster counts
  count = CountPredictions(all_data)

  # Save clustering results to JSON file
  if filename is not None:
    filepath = SaveAsJson(kmeans,
                          cluster_features,
                          all_data.columns.tolist(),
                          count,
                          filename,
                          save_path,
                          cluster_features)
  else:
    filepath = None
  return filepath, all_data


#######################  CLUSTER PROCESSING  ############################
# Calculate the entropy of discrete (categorical) and continuous data.

def ComputeContinuousEntropy(data, min, step):
  """Compute the entropy of a floating point distribution.  Bin the data, with
  the given step size, and then compute the discrete entropy."""
  idata = ((np.asarray(data)-min)/step).astype(int)
  return ComputeDiscreteEntropy(idata)


def ComputeDiscreteEntropy(idata):
  # https://stackoverflow.com/questions/4746812/count-the-multiple-occurrences-in-a-set
  d = Counter(idata)
  p = np.asarray(list(d.values())).astype(float) / len(idata)
  return -np.sum(p * np.log2(p))
