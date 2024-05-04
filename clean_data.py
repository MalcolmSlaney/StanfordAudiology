# Enhance the FM clinical data dump by
#   Replacing the MRN with a HMAC code
#   Classifying the type of hearing loss (sensorial neural, conductive, mixed)
#   Calculate the speech intelligibiity index
#   Enhance the dataset with some pure-tone averages
#   Add cluster IDs for a number of different cluster counts
#   Flag the patients with multiple visits
# and then write out the new data.

# To Run:
# python3 clean_data.py --input /Users/malcolm/Downloads/3-27-2024\ raw\ data\ no\ cleaning.csv \
#  --hmac_key ReplaceMe --output 3-27-2024-clean-enhanced-data.csv

import os
from typing import List, Union

from absl import app
from absl import flags
import hmac
import hashlib
import numpy as np
import pandas as pd
from scipy import interpolate

import clusters
from speech_intelligibility_index import sii

# Default to sha3_384, based on this article
# https://crypto.stackexchange.com/questions/68307/what-is-the-difference-between-sha-3-and-sha-256

def create_hmac(data, key, hash_type=hashlib.sha3_384):
  digest = hmac.new(bytes(key, 'UTF-8'),
                    bytes(str(data), 'UTF-8'), hashlib.sha256)
  return digest.hexdigest()


def replace_mrn(df, key: str = 'ReplaceMe') -> pd.DataFrame:
  df['Patients::HMAC'] = df.apply(lambda row: create_hmac(row['Patients::MRN'], key), axis=1)
  df = df.drop(columns='Patients::MRN', errors='ignore')
  df = df.drop(columns='HMAC_code', errors='ignore')
  return df


def classify_hearing_loss(df: pd.DataFrame):
  # Assign result to a temporary dataframe, since classifier creates lots of 
  # temporary column names.

  orig_df = df.copy()
  new_df = clusters.HLossClassifier(df)
  df = orig_df
  df['R_Type_HL_Mod'] = new_df['R_Type_HL_Mod']
  df['R_Type_HL_HF'] = new_df['R_Type_HL_HF']
  df['R_Type_HL_All'] = new_df['R_Type_HL_All']

  df['L_Type_HL_Mod'] = new_df['L_Type_HL_Mod']
  df['L_Type_HL_HF'] = new_df['L_Type_HL_HF']
  df['L_Type_HL_All'] = new_df['L_Type_HL_All']

  return df


# From: https://colab.research.google.com/drive/1bAnDpUUx5-BYkL3l3ukNVVnsLzEhy-ds#scrollTo=v6Dd6A2-UWG3

default_audiogram_freqs = [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]

def sii_from_audiogram(
    audiogram_samples: Union[List[float], np.ndarray],
    audiogram_freqs: Union[List[float], np.ndarray] = default_audiogram_freqs) -> float:
    """Compute the speech intelligibility index from an audiogram

    Args:
      audiogram_samples: Hearing loss in db at frequencies corresponding to the
        next argument
      audiogram_freqs: Frequencies (Hz) where audiogram is measured
    Returns:
      a float, the corresponding speech intelligibility index
    """
    audiogram_samples = np.asarray(audiogram_samples)
    audiogram_freqs = np.asarray(audiogram_freqs)
    assert audiogram_samples.ndim == 1
    assert audiogram_freqs.ndim == 1
    assert audiogram_samples.shape[0] == audiogram_freqs.shape[0]

    # Interpolate from the supplied audiogram frequencies to the critical
    interp_func = interpolate.interp1d(audiogram_freqs, audiogram_samples,
                                       kind='quadratic',
                                       fill_value='extrapolate')
    critical_band_hl = interp_func(sii.mid_band_freqs)

    [ssl, nsl, hearing_threshold] = sii.input_5p1(ssl='normal')
    return sii.sii(ssl=ssl, nsl=nsl, hearing_threshold=critical_band_hl)


def sii_from_df(df, ear='R'):
  freqs = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]
  names = [f'{ear}{f}' for f in freqs]
  values = df[names].values
  names_values = list(zip(names, values))
  try:
    good_names_values = [nv for nv in names_values if np.isfinite(nv[1])]
    if len(good_names_values) > 2:
      freqs, values = zip(*good_names_values)
      freqs = [float(f[1:]) for f in freqs]

      r_sii = sii_from_audiogram(values, freqs)
    else:
      r_sii = np.nan
  except:
    r_sii = np.nan
  return r_sii


def calculate_all_sii(df: pd.DataFrame):
  df['R_SII'] = df.apply(lambda df: sii_from_df(df, 'R'), axis=1)
  df['L_SII'] = df.apply(lambda df: sii_from_df(df, 'L'), axis=1)
  return df


def pta_summary(df: pd.DataFrame, 
                freqs: List[float], 
                ear: str ='R') -> np.ndarray:
  col_list = [f'{ear}{f}' for f in freqs]
  data = df[col_list]
  return np.mean(data.values, axis=1)

def all_pta_summaries(df: pd.DataFrame) -> pd.DataFrame:
  pta_freqs = [500, 1000, 2000, 4000]
  conv_freqs = [500, 1000, 2000]
  hf_freqs = [1000, 2000, 4000]

  df['L_PTA'] = pta_summary(df, pta_freqs, 'L')
  df['L_Conv_PTA'] = pta_summary(df, conv_freqs, 'L')
  df['L_HF_PTA'] = pta_summary(df, hf_freqs, 'L')

  df['R_PTA'] = pta_summary(df, pta_freqs, 'R')
  df['R_Conv_PTA'] = pta_summary(df, conv_freqs, 'R')
  df['R_HF_PTA'] = pta_summary(df, hf_freqs, 'R')

  return df


def add_cluster_ids(df: pd.DataFrame, 
                    cluster_counts: List[int] = [6, 8, 10, 12],
                    cluster_dir='StanfordAudiology/ClusterData_v1'):
  for cluster_count in cluster_counts:
    cluster_data_filename = os.path.join(cluster_dir, 
                                         f'Cluster{cluster_count:02}Way.json')
    kmeans, feature_labels, all_features = clusters.LoadFromJson(cluster_data_filename)
    kmeans._n_threads = 1
    kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(np.double)

    good_data = df[feature_labels].dropna()
    cluster_ids = clusters.KMeansPredictions(kmeans, good_data)
    feature_name = f'Cluster{cluster_count:02}Way'
    good_data[feature_name] = cluster_ids.astype(int)
    df[feature_name] = good_data[feature_name]  
  return df


def label_duplicates(df: pd.DataFrame) -> pd.DataFrame:
  duplicate_name = 'MultipleVisits'
  df[duplicate_name] = False
  df.loc[df['Patients::HMAC'].duplicated(keep=False), 'MultipleVisits'] = True
  return df


FLAGS = flags.FLAGS
flags.DEFINE_string('input',
                    '/Users/malcolm/Downloads/3-27-2024 raw data no cleaning.csv',
                    'Input CSV filename')
flags.DEFINE_string('output',
                    'All_Clean_Audiology_Data.csv',
                    'Output CSV filename')
flags.DEFINE_string('hmac_key',
                    None,
                    'Key to use when hashing the MRN.',
                    required=True)
flags.DEFINE_string('cluster_dir', 'ClusterData_v1',
                    'Where to find the pretrained cluster json data')
flags.DEFINE_string('convert_mrns', None, 
                    'Which file to read MRNs from to convert to hashes')

def main(argv):
  if FLAGS.convert_mrns:
    # Just convert a list of MRNs (in a file) into their hash IDs.
    with open(FLAGS.convert_mrns, 'r') as fp:
      print(fp.readline().strip().replace('MRN', 'HMAC'))  # Replace the column header
      for line in fp:
        line = line.strip()
        print(create_hmac(line, FLAGS.hmac_key, hash_type=hashlib.sha3_384))
  return

  df = pd.read_csv(FLAGS.input,
                   on_bad_lines='warn',
                   na_values=['NR', '\n',
                              '? * Line 1, Column 1\n  Syntax error: value, '
                              'object or array expected.\n* Line 1, Column 1\n'
                              '  A valid JSON document must be either an array '
                              'or an object value.']) 
  print('Column names:', df.columns.tolist())

  df = replace_mrn(df, FLAGS.hmac_key)
  df = classify_hearing_loss(df)
  df = calculate_all_sii(df)
  df = all_pta_summaries(df)
  df = add_cluster_ids(df, cluster_dir=FLAGS.cluster_dir)
  df = label_duplicates(df)
  df.to_csv(FLAGS.output)

  num_multiples = len(set(df.loc[df.MultipleVisits]['Patients::HMAC']))
  num_patients = len(set(df['Patients::HMAC']))
  num_visits = len(df['Patients::HMAC'])
  print(f'Total # Records: {num_visits}, # Patients: {num_patients}, '
        f'# Multiple Visit Patients: {num_multiples}')


if __name__ == '__main__':
  app.run(main)