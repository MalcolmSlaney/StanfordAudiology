"""De-identification of audiological data using HMAC code."""

import hmac
import hashlib
import pandas as pd


def create_hmac(data, key, hash_type=hashlib.sha3_384):
  """Create a HMAC Code using unique key."""
  digest = hmac.new(bytes(key, 'UTF-8'),
                    bytes(str(data), 'UTF-8'), hashlib.sha256)
  return digest.hexdigest()


def replace_mrn(df, key: str = 'ReplaceMe') -> pd.DataFrame:
  """Replace MRN (PHI) with HMAC Code - True De-identification."""
  df['Patients::HMAC'] = df.apply(
    lambda row: create_hmac(row['Patients::MRN'], key), axis=1)
  df = df.drop(columns='Patients::MRN', errors='ignore')
  df = df.drop(columns='HMAC_code', errors='ignore')
  return df