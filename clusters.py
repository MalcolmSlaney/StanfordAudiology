
#Global variables/parameters

default_cluster_dir = '/content/drive/MyDrive/Stanford Audiology Models/Colab Notebooks/'

"""
A change is required in the spreadsheet_path. Open the spreadsheet mfb raw audio
and copy the link from the address bar. The gid code in the url is unique for each
of the user.
"""
spreadsheet_v1_path = 'https://docs.google.com/spreadsheets/d/119_-qrfzGXwV1YBUJdBzvtAQTZnl-xwN7hD9FK5SWfU/edit#gid=84023254'
# features_before = [ "AgeAtTestDate", "R250", "R500", "R1000", "R2000", "R3000", "R4000", "R6000", "R8000",
#               "L250", "L500", "L1000", "L2000", "L3000", "L4000", "L6000", "L8000", "RBone500",
#               "RBone1000", "RBone2000", "RBone4000", "LBone500", "LBone1000", "LBone2000", "LBone4000",
#               "MonSNR_Score_R", "Word_Rec_Score_R", "MonSNR_Score_L", "Word_Rec_Score_L" ]

duplicate_column_name_1: str = 'LBone2000'

#Golden cluster can be computed by calling CreateKMeans() with random_seed = 0

golden_cluster = {(69.764465, 17.85707): 'High flat',
 (22.426544, 19.861082): 'Low slope',
 (42.734375, 5.0265923): 'Mid flat',
 (35.734127, 46.22456): 'Mid slope',
 (9.56347, 2.5132303): 'Low flat',
 (52.65366, 47.10048): 'High slope'}

import numpy as np
import dataclasses
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import warnings
import joblib
import sklearn
from typing import List
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import json
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
import os
import datetime

from google.colab import auth
import gspread
from google.auth import default  #authenticatiing to google

def ImportSpreadsheet(path) -> List:
  """
  Function that imports the spreadsheet from the specified `path`
  Args:
    - path - Filepath to the spreadsheet [GCloud URL]
  Return:
    - rows - Each row of the spreadsheet is stored as a list. First row of the
      list contains feature names.
  """
  auth.authenticate_user()
  creds, _ = default()
  gc = gspread.authorize(creds)
  worksheet = gc.open_by_url(path).sheet1 #get_all_values gives a list of rows
  rows = worksheet.get_all_values()

  return rows

def ConvertToPanda(data: List[dataclasses.dataclass],
             feature_names: List) -> pd.DataFrame:
  """
  Converting the HL data at audiometric frequencies to pandas dataframe classify
  the type of hearing loss

  Args:
    - data: List data that has to be converted to a pandas dataframe
    - column_names: names of the columns of data
  Return:
    - df:  pandas.DataFrame
  """
  data = data[:]
  df = pd.DataFrame(data, columns = feature_names)
  return df

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

  Args:
    - df:  dataframe with HL measurements at audiometric frequencies

  Returns:
    - df: dataframe with HL classes as a new column
  """
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    #HFPTA
    df_hfpta_R  = df[['R1000', 'R2000', 'R4000']]
    df_hfpta_L  = df[['L1000', 'L2000', 'L4000']]
    HFPTA_R = np.array(np.nanmean(df_hfpta_R.to_numpy(),axis=1))
    HFPTA_L = np.array(np.nanmean(df_hfpta_L.to_numpy(),axis=1))
    df.loc[:,'R_HFPTA'] = HFPTA_R
    df.loc[:,'L_HFPTA'] = HFPTA_L

    # PTA - 500, 1000, 2000
    df_pta_R = df[['R500', 'R1000', 'R2000']]
    df_pta_L = df[['L500', 'L1000', 'L2000']]
    PTA_R = np.array(np.nanmean(df_pta_R.to_numpy(),axis=1))
    PTA_L = np.array(np.nanmean(df_pta_L.to_numpy(),axis=1))
    df.loc[:,'R_PTA'] = PTA_R
    df.loc[:,'L_PTA'] = PTA_L

    #PTA all -  500,1000,2000,4000
    df_pta_all_R = df[['R500', 'R1000', 'R2000', 'R4000']]
    df_pta_all_L = df[['L500', 'L1000', 'L2000', 'L4000']]
    PTA_all_R = np.array(np.nanmean(df_pta_all_R.to_numpy(),axis=1))
    PTA_all_L = np.array(np.nanmean(df_pta_all_L.to_numpy(),axis=1))
    df.loc[:,'R_PTA_All'] = PTA_all_R
    df.loc[:,'L_PTA_All'] = PTA_all_L

    #LFPTA -  250, 500,1000
    df_lfpta_R = df[['R500', 'R1000']]
    df_lfpta_L = df[['L500', 'L1000']]
    LFPTA_R = np.array(np.nanmean(df_lfpta_R.to_numpy(),axis=1))
    LFPTA_L = np.array(np.nanmean(df_lfpta_L.to_numpy(),axis=1))
    df.loc[:,'R_LFPTA'] = LFPTA_R
    df.loc[:,'L_LFPTA'] = LFPTA_L

    #UHFPTA - 2000, 4000, 80000
    df_uhfpta_R = df[['R2000', 'R4000', 'R8000']]
    df_uhfpta_L = df[['L2000', 'L4000', 'L8000']]
    UHFPTA_R = np.array(np.nanmean(df_uhfpta_R.to_numpy(),axis=1))
    UHFPTA_L = np.array(np.nanmean(df_uhfpta_L.to_numpy(),axis=1))
    df.loc[:,'R_UHFPTA'] = UHFPTA_R
    df.loc[:,'L_UHFPTA'] = UHFPTA_L

    #PT Bone conduction modeled
    df_pta_BC_mod_R = df[['RBone500', 'RBone1000', 'RBone2000']]
    df_pta_BC_mod_L = df[['LBone500', 'LBone1000', 'LBone2000']]
    PTA_BC_mod_R = np.array(np.nanmean(df_pta_BC_mod_R.to_numpy(), axis=1))
    PTA_BC_mod_L = np.array(np.nanmean(df_pta_BC_mod_L.to_numpy(), axis=1))
    df.loc[:,'R_PTA_BC_Mod'] = PTA_BC_mod_R
    df.loc[:,'L_PTA_BC_Mod'] = PTA_BC_mod_L

    #HFPTA Bone conduction modeled
    df_hfpta_BC_mod_R = df[['RBone1000', 'RBone2000','RBone4000']]
    df_hfpta_BC_mod_L = df[['LBone1000', 'LBone2000','LBone4000']]
    HFPTA_BC_mod_R = np.array(np.nanmean(df_hfpta_BC_mod_R.to_numpy(),axis=1))
    HFPTA_BC_mod_L = np.array(np.nanmean(df_hfpta_BC_mod_L.to_numpy(),axis=1))
    df.loc[:,'R_HFPTA_BC_Mod'] = HFPTA_BC_mod_R
    df.loc[:,'L_HFPTA_BC_Mod'] = HFPTA_BC_mod_L

    #BC average of 500, 1, 2, 4
    df_hfpta_BC_avg_R = df[['RBone500','RBone1000', 'RBone2000','RBone4000']]
    df_hfpta_BC_avg_L = df[['LBone500','LBone1000', 'LBone2000','LBone4000']]
    HFPTA_BC_avg_R = np.array(np.nanmean(df_hfpta_BC_avg_R.to_numpy(),axis=1))
    HFPTA_BC_avg_L = np.array(np.nanmean(df_hfpta_BC_avg_L.to_numpy(),axis=1))
    df.loc[:,'R_PTA_BC_All'] = HFPTA_BC_avg_R
    df.loc[:,'L_PTA_BC_All'] = HFPTA_BC_avg_L

    # new ABGap
    df.loc[:,'R_PTA_ABGap'] = df['R_PTA'] - df['R_PTA_BC_Mod']
    df.loc[:,'R_HFPTA_ABGap'] = df['R_HFPTA'] - df['R_HFPTA_BC_Mod']
    df.loc[:,'R_PTA_All_ABGap'] = df['R_PTA_All'] - df['R_PTA_BC_All']

    df.loc[:,'L_PTA_ABGap'] = df['L_PTA'] - df['L_PTA_BC_Mod']
    df.loc[:,'L_HFPTA_ABGap'] = df['L_HFPTA'] - df['L_HFPTA_BC_Mod']
    df.loc[:,'L_PTA_All_ABGap'] = df['L_PTA_All'] - df['L_PTA_BC_All']

    #HL Type
    #Right
    # using the new Modeled BC PTA of 5, 1, 2
    conditions_1 = [
        (df['R_PTA_BC_Mod'] < 25.1) & (df['R_PTA_ABGap'] >= 10) & (df['R_PTA'] > 25),
        (df['R_PTA_ABGap'] < 10) & (df['R_PTA'] > 25),
        (df['R_PTA_BC_Mod'] > 25) & (df['R_PTA_ABGap'] >= 10) & (df['R_PTA'] > 25)
                   ]
    values = ['Conductive','SNHL','Mixed']
    df['R_Type_HL_Mod']= np.select(conditions_1, values)
    df.loc[df["R_Type_HL_Mod"] == '0' , "R_Type_HL_Mod"] = 'Normal'

    # HFPTA of 1
    conditions_2 = [
        (df['R_HFPTA_BC_Mod'] < 25.1) & (df['R_HFPTA_ABGap'] >= 10) & (df['R_HFPTA'] > 25),
        (df['R_HFPTA_ABGap'] < 10) & (df['R_HFPTA'] > 25),
        (df['R_HFPTA_BC_Mod'] > 25) & (df['R_HFPTA_ABGap'] >= 10) & (df['R_HFPTA'] > 25)
                   ]

    values = ['Conductive','SNHL','Mixed']
    df['R_Type_HL_HF'] = np.select(conditions_2, values)
    df.loc[df["R_Type_HL_HF"] == '0' , "R_Type_HL_HF"] = 'Normal'

    # # PTA of 500 1 2 4
    conditions_3 = [
        (df['R_PTA_BC_All'] < 25.1) & (df['R_PTA_All_ABGap'] >= 10) & (df['R_PTA_All'] > 25),
        (df['R_PTA_All_ABGap'] < 10) & (df['R_PTA_All'] > 25),
        (df['R_PTA_BC_All'] > 25) & (df['R_PTA_All_ABGap'] >= 10) & (df['R_PTA_All'] > 25)
                   ]
    values = ['Conductive','SNHL','Mixed']
    df['R_Type_HL_All'] = np.select(conditions_3, values)
    df.loc[df["R_Type_HL_All"] == '0' , "R_Type_HL_All"] = 'Normal'
  return df

def HLPlot(df: pd.DataFrame, title = None):
  """
  This function displays a scatter plot of Avg air conduction vs Avg bone conduction
  threshold coloured by 4 hearing loss classes:
    - Normal
    - SNHL
    - Conductive
    - Mixed
  Args:
    - df: dataframe computed by function HL_loss_classification()
    - title: title of the plot (incase when needed)

  Displays:
      a scatter plot in the current figure.

  """
  df_normal = df.loc[df['R_Type_HL_All'] == 'Normal']
  df_snhl = df.loc[df['R_Type_HL_All'] == 'SNHL']
  df_conductive = df.loc[df['R_Type_HL_All'] == 'Conductive']
  df_mixed = df.loc[df['R_Type_HL_All'] == 'Mixed']

  plt.scatter(df_normal['R_PTA_BC_All'], df_normal['R_PTA_All'], color = 'r',alpha = 0.5, label = 'Normal')
  plt.scatter(df_snhl['R_PTA_BC_All'], df_snhl['R_PTA_All'], color = 'b',alpha = 0.5, label = 'SNHL')
  plt.scatter(df_conductive['R_PTA_BC_All'], df_conductive['R_PTA_All'], alpha = 0.5, color = 'g', label = 'Conductive')
  plt.scatter(df_mixed['R_PTA_BC_All'], df_mixed['R_PTA_All'], color = 'black',alpha = 0.5, label = 'Mixed')

  plt.xlabel('Average Bone Conduction Threshold')
  plt.ylabel("Average Air Conduction Threshold")
  plt.title(title)
  plt.xlim(-20, 100)
  plt.ylim(-20, 130)
  plt.legend()
  plt.show()

def RenameDuplicateColumns(row_list: List,
                           column_name: str) -> List:
  """
  Function to rename a duplicate column name.

  Args:
    - row_list: A list containing each row_list of the spreadsheet
    - column_name: Name of the duplicate column

  Returns:
   - A list containing no duplicate for the specified column_name
  """
  indices = [i for i in range(len(row_list[0])) if row_list[0][i] == column_name]
  if len(indices) > 1:
    row_list[0][indices[1]] = column_name+'Dupe'

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

def ConvertToNumerical(sd: List[dataclasses.dataclass],
                  field_list: List[str],
                  type=np.float32) -> np.ndarray:
  """
  Collect the data from each field listed in list and convert the string value
  to a numerical value

  Args:
    - sd: A list of Stanford dataclass objects (holding the audiogram data)
    - field_list: A list of column names to extract
    - type: The desired data type, should be a NP type or object.

  Returns:
    - numpy array of desired datatype (NP type or object)
  """
  data = np.zeros((len(sd), len(field_list)), dtype=type)
  for i, r in enumerate(sd):
    for j, f in enumerate(field_list):
      d = r.__getattribute__(f)

      if type==np.float32 or type==np.float64:
        if d == '' or d == 'NR':
          d = np.nan
        else:
          try:
            d = float(d)
          except:   # Last resort
            d = np.nan

      data[i, j] = d
  return data

def MakeDataClass(column_names: List,
                  rows: List, features: List) -> List:
  """
  This function creates a new dataclass (StanfordData) object for each row of
  data, and then returns a list of these objects.

  Args:
    - column_names: A list of strings that represents the names of the columns
      in the data.
    - rows: A list of lists of strings that represents the rows of data.
    - features: A subset of column_names that is needed for further processing

  Returns:
    - A list of StanfordData objects, where each object represents a row of data,
    and no column names are present.
  """
  column_names = ConvertSpaces(column_names) #To convert spaces in column names to _
  StanfordData = dataclasses.make_dataclass('stanford_data', column_names)
  all_stanford_data = [StanfordData(*rows[i]) for i in range(1, len(rows))]

  #To find the datapoints with apropriate age limit
  age_data =  ConvertToNumerical(all_stanford_data, ['AgeAtTestDate'])

  good_age_rows,_ = np.where((age_data >=0) & (age_data <=100))
  good_rows = [all_stanford_data[i] for i in good_age_rows]
  data = ConvertToNumerical(good_rows, features)

  return data

def CheckBCandAC(data: pd.DataFrame, threshold: int = 100) -> pd.DataFrame:
  """
  A function to drop the rows where Bone Conduction is greater than Air 
  Conduction by atleast 10dB

  Args:
  - data: A pandas dataframe on which HLossClassifier() function is run.
  - threshold: The minimum difference between air conduction and bone conduction
              loss.
  Return:
  - data: A pandas dataframe where the rows with bone conduction loss > air conduction 
  loss are dropped
  """

  data = data.drop(data.loc[data['R_PTA_BC_All'] - data['R_PTA_All'] >= threshold].index)
  return data

def CreateKMeans(n: int,
                 data: pd.DataFrame,
                 random_state:int = 0,
                 max_iter: int = 1000,
                 n_init: int = 10) -> sklearn.cluster._kmeans.KMeans:
  if n <= len(data):
    kmeans = KMeans(n_clusters = n,
                    init ='k-means++',
                    max_iter = max_iter,
                    n_init = n_init,
                    random_state = random_state)
    kmeans.fit(data) #To compute cluster centres
    return kmeans

def PlotClusterCenters(labels: List[str],
                       kmeans: sklearn.cluster._kmeans.KMeans,
                       cluster_label: dict,
                       n: int):
  """
  This function plots the clusters created using create_k_means

  Args:
    - labels: Labels for x-axis
    - kmeans: kmeans clusters computed using scikit learn package
    - cluster_label: A dictionary that assigns predefined labels to clusters
    - n: Number of clusters
  """
  color = plt.cm.rainbow(np.linspace(0, 1, 6))

  """

  Labels are of the form string(XNNN) where X is L or R for left and right ears,
  and NNN is the frequemcy. Hence, we extract only int(NNN) from the labels
  """
  conv_labels = [int(value[1:]) for value in labels]

   #Check the shape of the cluster centers array
  if kmeans.cluster_centers_.ndim == 0:
    raise ValueError("The cluster centers array must be at least 1-dimensional")

  #Plot
  for i, c in zip(range(n), color):
    plt.semilogx(conv_labels, kmeans.cluster_centers_.T[:,i], label = cluster_label[i])
    plt.legend()


  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Mean Hearing Loss (dB)')
  plt.title('6 Clusters formed with respect to the standard frequencies')
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
  n = 4 #Number of clusters

  # Generate data using make_blobs
  centers = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])  # Centers of the Gaussian blobs
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
  print("Cluster Centers Verification (|y| < 0.1):", any(valid_centers))
  print("Clusters Verification (x in {0, 1, 2, 3}):", list(sorted(df['Cluster'].unique())) == [0,1,2,3])

def KMeansPredictions(kmeans: sklearn.cluster._kmeans.KMeans,
                      data: pd.DataFrame,
                      cluster_label: dict):
  """
  Predicts the cluster labels for the given data using the given KMeans model.

  Args:
    - kmeans: The KMeans model to be used for predicting the cluster labels.
    - data: The data to be clustered.
    - cluster_label: A dictionary that maps cluster labels to their names.
  """
  p = kmeans.predict(data)
  data['predictions'] = p
  return data

def CountPredictions(data: pd.DataFrame,
                      cluster_label: dict) -> int :
  """
  Counts the number of patients in each cluster.

  Args:
      data: The DataFrame containing the HL classified data.
      cluster_label: A dictionary that maps cluster labels to their names.

  Returns:
      A dictionary that maps cluster labels to the number of patients in each cluster.
  """
  predictions = data['predictions']
  labels = np.unique(predictions)
  count = {}
  for i in labels:
    count_temp = len(data[data['predictions'] == i])
    count[cluster_label[i]] = count_temp
  return count

def AssignClusterLabelsToDF(data: pd.DataFrame,
                          cluster_label: dict) -> pd.DataFrame:

  """Assigns cluster labels to the data.

  Args:
    data: The data to assign cluster labels to.
    cluster_label: A dictionary mapping cluster IDs to cluster labels.

  Returns:
    The data with the cluster labels assigned.
  """
  data['cluster_labels'] = float('nan')
  for i in list(cluster_label.keys()):
    data.loc[data['predictions'] == i, 'cluster_labels'] = cluster_label[i]
  return data


def SaveAsJson(cluster_labels: dict,
               kmeans: sklearn.cluster._kmeans.KMeans,
               features_before: List,
               features_after: List,
               nClusterPoints: dict,
               filename,
               path: str = default_cluster_dir) -> str:
  """
  Save the KMeans clustering results as a JSON file.

  Args:
    - path: The path of the google directory where the JSON file will be saved.
    - cluster_labels: A dictionary mapping cluster labels to the corresponding data points.
    - kmeans: The KMeans model that was used to cluster the data.
    - features_before: The list of features that were used to cluster the data.
    - features_after: The list of features that were obtained after performing HL classification.
    - filename: Name of the file
    - nClusterPoints: A dictionary mapping cluster labels to the number of data points in each cluster.

  Returns:
    The path to the JSON file that was saved.
  """

  data = {
        "cluster_centers": kmeans.cluster_centers_.tolist(),
        "random_state": kmeans.random_state,
        "n_init": kmeans.n_init,
        "n_cluster": kmeans.n_clusters,
        "max_iter": kmeans.max_iter,
        "features_before": features_before,
        "features_after": features_after,
        "cluster_labels": cluster_labels,
        "n_cluster_points": nClusterPoints,
        "date_time": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        }
  filepath = os.path.join(path + filename + ".json")
  with open(filepath, "w") as file:
      json.dump(data, file, indent=4)

  return filepath

def change_key_values_to_integer(dictionary):
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

def LoadFromJson(
                path: str = default_cluster_dir,
                ) -> tuple[KMeans, List[str]]:

  """
  Load the KMeans clustering results from a JSON file.

  Args:
    path: The path of the google directory where the JSON file will be loaded from.

  Returns:
    A tuple of the KMeans model, features before HL classification, cluster labels
    and the list of features after HL classification.
  """

  with open(path, "r") as file:
    info = json.load(file)

  cluster_centers = np.array(info["cluster_centers"])
  random_state = info["random_state"]
  cluster_labels = change_key_values_to_integer(info["cluster_labels"])
  n_init = info["n_init"]
  n_cluster = info['n_cluster']
  max_iter = info['max_iter']
  features_before = info['features_before']
  features_after = info['features_after']
  date_time = info['date_time']

  #Initialise a kmeans object
  kmeans = KMeans()
  kmeans.cluster_centers_ = cluster_centers.astype(np.float32)
  kmeans.random_state = random_state
  kmeans.n_init = n_init
  kmeans.n_clusters = n_cluster
  kmeans.max_iter = max_iter
  kmeans._n_threads = _openmp_effective_n_threads()

  return kmeans, features_before, cluster_labels, features_after

def ReadPreprocessData(duplicate_column_name: str = duplicate_column_name_1,
              spreadsheet_path: str = spreadsheet_v1_path) -> np.ndarray:

  """
    Clean and transform data from a spreadsheet into a pandas DataFrame.

    Parameters:
        features (List[str], optional): A list of strings representing the names of the features to include in the DataFrame.
                                        Defaults to the value of `features_before`.
        duplicate_column_name (str, optional): The name of the duplicate column to handle in the data.
                                               Defaults to the value of `duplicate_column_name_1`.
        spreadsheet_path (str, optional): The file path of the spreadsheet containing the data to process.
                                          Defaults to the value of `spreadsheet_v1_path`.

    Returns:
        data: A numpy array containing the cleaned and transformed data.
        features: A list of column names 
  """
  
  rows = ImportSpreadsheet(spreadsheet_path)
  rows = RenameDuplicateColumns(rows, duplicate_column_name)
  features = ConvertSpaces(rows[0])

  data = MakeDataClass(features, rows)


  return (data, features)

def euclidean_distance(centroid1, centroid2) -> float:
  """
    Calculate the Euclidean distance between two cluster centroids represented as mean and slope.

    Parameters:
    centroid1 : The first centroid
    centroid2 : The second centroid

    Returns:
    float: The Euclidean distance between the two centroids.

  """
  return np.linalg.norm(np.array(centroid1) - np.array(centroid2))

def SlopeandMean(kmeans: sklearn.cluster._kmeans.KMeans):

  """
  Calculate the slopes and means of the cluster centers obtained from a k-means clustering algorithm.

  Parameters:
  kmeans : sklearn.cluster._kmeans.KMeans
      The result of a k-means clustering algorithm containing cluster centers.

  Returns:
  list of tuples
      A list of tuples, where each tuple contains the mean and slope corresponding to a cluster center.
  """

  slopes = kmeans.cluster_centers_.T[7,:] - kmeans.cluster_centers_.T[0,:]
  means = np.mean(kmeans.cluster_centers_, axis = 1)
  slope_mean = [(m, s) for m, s in zip(means, slopes)]
  return slope_mean

def AssignClusterLabels(kmeans, ref_cluster: dict = golden_cluster ):

  """
    Assign cluster labels to centroids based on their proximity to the centroids in the `golden_cluster` dictionary.

    Parameters:
    kmeans : object
        The result of a k-means clustering algorithm containing centroids and cluster assignments.

    ref_cluster : dict, optional
        A dictionary representing the `golden_cluster` centroids. It should have centroids as keys and cluster labels as values.
        The default value is `golden_cluster`, which may represent some pre-defined "golden" cluster centroids.
        
    Returns:
        A dictionary containing the updated cluster labels for each centroid in `kmeans`. The keys represent the index of the centroid in `kmeans`,
        and the values are the corresponding cluster labels from the `golden_cluster` dictionary.
  """
  
  clusters_slope_mean = SlopeandMean(kmeans)
  new_labels = {}
  golden_cluster_centroids = list(golden_cluster.keys())
  golden_cluster_centroids

  for mean_slope, i in zip(clusters_slope_mean, range(len(clusters_slope_mean))):
        distance_mean = [euclidean_distance(mean_slope[0], centroid[0]) for centroid in golden_cluster_centroids]
        distance_slope = [euclidean_distance(mean_slope[1], centroid[1]) for centroid in golden_cluster_centroids]
        index_mean = np.argmin(distance_mean)
        index_slope = np.argmin(distance_slope)

        if index_mean == index_slope:
          index = index_mean
          new_labels[i] = golden_cluster[golden_cluster_centroids[index]]

  return new_labels


def CreateClusterV1(filename: str,
                    duplicate_column_name: str = duplicate_column_name_1,
                    labels: List[str] = ['R250',	'R500',	'R1000',	'R2000',	'R3000',	'R4000',	'R6000',	'R8000'],
                    spreadsheet_path: str = spreadsheet_v1_path,
                    save_path: str = default_cluster_dir,
                    cluster_labels: dict = {4: 'Low flat',
                                            2: 'Mid flat',
                                            0: 'High flat',
                                            1: 'Low slope',
                                            3: 'Mid slope',
                                            5: 'High slope'},
                    random_state: int = 0,
                    n: int = 6,
                    max_iter: int = 1000,
                    n_init: int = 10):
  """
    Creates clusters using the K-means algorithm and saves the results as a JSON file.

    Parameters:
    - filename (str): The name of the JSON file to be saved.
    - features (List[str]): A list of column names in the dataset to be used as features for clustering.
    - duplicate_column_name (str): The name of the duplicate column to be renamed.
    - labels (List[str], optional): A list of column names to be used as labels for evaluation. 
                                    Default is ['R250', 'R500', 'R1000', 'R2000', 'R3000', 'R4000', 'R6000', 'R8000'].
    - spreadsheet_path (str, optional): The path to the spreadsheet file. Default is spreadsheet_v1_path.
    - save_path (str, optional): The directory where the JSON file will be saved. Default is default_cluster_dir.
    - cluster_labels (dict, optional): A dictionary mapping cluster indices to cluster labels. 
                                      Default is {4: 'Low flat', 2: 'Mid flat', 0: 'High flat', 
                                      1: 'Low slope', 3: 'Mid slope', 5: 'High slope'}.
    - random_state (int, optional): The random seed to be used for reproducible results. Default is 0.
    - n (int, optional): The number of clusters to be created. Default is 6.
    - max_iter (int, optional): The maximum number of iterations for the K-means algorithm. Default is 1000.
    - n_init (int, optional): The number of times the K-means algorithm will be run with different centroid seeds. 
                              Default is 10.

    Returns:
    - str: The path to the saved JSON file.

    Raises:
    - Any relevant exceptions that could occur during the execution of the function.

    Example usage:
    >>> CreateClusterV1('clusters.json', ['feature1', 'feature2'], 'duplicate_column')

    Note:
    - This function assumes the existence of other necessary functions 
      like ImportSpreadsheet, RenameDuplicateColumns, MakeDataClass, ConvertToPanda, 
      HLossClassifier, CreateKMeans, KMeansPredictions, AssignClusterLabels, CountPredictions, and SaveAsJson.
    - Make sure to import the required modules and define the necessary functions before using this function.
    """

  rows = ImportSpreadsheet(spreadsheet_path)

  rows = RenameDuplicateColumns(rows, duplicate_column_name)
  data = MakeDataClass(rows[0], rows, features)
  df = ConvertToPanda(data, features)
  df = HLossClassifier(df)

  hl = df.dropna(subset = labels)
  hl = hl[labels]

  kmeans = CreateKMeans(n,hl,random_state,max_iter,n_init)
  hl = KMeansPredictions(kmeans, hl, cluster_labels)
  df['predictions'] = hl['predictions']

  data = AssignClusterLabels(df, cluster_labels)
  count = CountPredictions(hl, cluster_labels)
  SaveAsJson(cluster_labels,
           kmeans,
           features,
           df.columns.tolist(),
           count,
           filename,
           save_path)

  return save_path+filename