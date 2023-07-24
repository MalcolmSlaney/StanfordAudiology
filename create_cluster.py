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
from clusters import *

def CreateClusterV1(filename: str,
                    features: List[str] = [
        "AgeAtTestDate",
        "R250",
        "R500",
        "R1000",
        "R2000",
        "R3000",
        "R4000",
        "R6000",
        "R8000",
        "L250",
        "L500",
        "L1000",
        "L2000",
        "L3000",
        "L4000",
        "L6000",
        "L8000",
        "RBone500",
        "RBone1000",
        "RBone2000",
        "RBone4000",
        "LBone500",
        "LBone1000",
        "LBone2000",
        "LBone4000",
        "MonSNR_Score_R",
        "Word_Rec_Score_R",
        "MonSNR_Score_L",
        "Word_Rec_Score_L"
    ],
                    duplicate_column_name: str = 'LBone200',
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
