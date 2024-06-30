from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from sklearn.linear_model import Ridge, LinearRegression
from urllib.request import DataHandler

# from google.colab import drive
# drive.mount('/content/drive')

data_folder = '/content/drive/MyDrive/Research/ABR Analysis Project/Bidelman Data/'


def extract_abr(subject, period, run, bandpass_lo_freq=90, 
                bandpass_high_freq=2000, print_info=False,
                data_folder=data_folder):
  folder_labels = [str(bandpass_lo_freq), str(bandpass_high_freq), '50ms/']
  folder_path = data_folder + '-'.join(folder_labels)

  subject_labels = ['subject' + str(subject), 'period' + str(period), 
                    'run' + str(run), str(bandpass_lo_freq), 
                    str(bandpass_high_freq)]
  filename = '-'.join(subject_labels) + '.mat'

  EEG = loadmat(folder_path + filename)

  if EEG['data'].ndim != 3:
    raise ValueError("EEG data array must have 3 dimensions")

  data = np.array(EEG['data'][0, :, :])

  if print_info:
    print(subject_labels)
    print("number of time samples = ", data.shape[0])
    print("number of trials = ", data.shape[1])

  return data


def compute_energy(abr_data):
  energy = np.sum(abr_data ** 2, axis=0) # sum across time
  assert len(energy) == abr_data.shape[1]
  return energy


def compute_rms(abr_data):
  energy = compute_energy(abr_data)
  rms = np.sqrt(energy / len(energy))
  assert len(rms) == abr_data.shape[1]
  return rms


def reject_artifacts(abr_data, variance_percentile=95, 
                     max_magnitude_percentile=95, rms_percentile=95):
  """
  This function performs 3 artifact rejections in conjuction:
  1. Removing trials with variance above the percentile
  2. Removing trials with maximum magnitude (absolute amplitude) above the percentile
  3. Removing trials with RMS above the percentile
  """
  # print("original data shape:", abr_data.shape)
  variances = np.var(abr_data, axis=0) # across time per trial
  variance_threshold = np.percentile(variances, variance_percentile)
  variance_mask = (variances <= variance_threshold)

  max_magnitudes = np.max(np.abs(abr_data), axis=0) # across time per trial
  max_magnitude_threshold = np.percentile(max_magnitudes, 
                                          max_magnitude_percentile)
  max_magnitude_mask = (max_magnitudes <= max_magnitude_threshold)

  rms = compute_rms(abr_data)
  rms_threshold = np.percentile(rms, rms_percentile)
  rms_mask = (rms <= rms_threshold)

  artifact_mask = variance_mask & max_magnitude_mask & rms_mask
  filtered_abr_data = abr_data[:, artifact_mask]
  # print("data shape after rejecting artifacts:", filtered_abr_data.shape)
  return filtered_abr_data


def remove_offset(abr_data):
  filtered_data = np.array([abr_data[:, i] - np.mean(abr_data[:, i]) 
                            for i in range(abr_data.shape[1])]).T
  assert filtered_data.shape[0] == abr_data.shape[0]
  return filtered_data


def compute_covariance_per_trial(abr_data_trial, abr_data_mean):
  inner_product = np.dot(abr_data_trial - np.mean(abr_data_trial), 
                         abr_data_mean - np.mean(abr_data_mean))
  assert isinstance(inner_product, float)
  num_time_points = len(abr_data_trial)
  return inner_product / num_time_points

def compute_covariance_with_mean(abr_data):
  abr_data_mean = np.mean(abr_data, axis=1) # across trial per time
  covariance = np.array([compute_covariance_per_trial(abr_data[:, i], 
                                                      abr_data_mean) for i in range(abr_data.shape[1])]).T
  assert len(covariance) == abr_data.shape[1]
  return covariance


def compute_d_prime(signal, noise):
  return np.abs(np.mean(signal) - np.mean(noise)) / np.sqrt(
    (np.std(signal) ** 2 + np.std(noise) ** 2) / 2)


def plot_subject(subject, offset_removal=False, artifact_rejection=False, 
                 print_info=False):
  total_periods = 4
  total_runs = 2

  fig = plt.figure(figsize=(12, 6))
  fig.suptitle("Subject 2")

  for period in range(1, total_periods + 1):
    for run in range(1, total_runs + 1):
      plot_title = "Period " + str(period) + " Run " + str(run)
      abr_data = extract_abr(subject, period, run, print_info=print_info)

      if offset_removal:
        abr_data = remove_offset(abr_data)
      if artifact_rejection:
        abr_data = reject_artifacts(abr_data)

      abr_mean = np.mean(abr_data, axis=1) # across trials
      abr_std = np.std(abr_data, axis=1) # across trials

      if run == 1:
        subplot_num = period
      else:
        subplot_num = period + total_periods

      plt.subplot(total_runs, total_periods, subplot_num)
      times = np.linspace(0, 50, abr_data.shape[0])
      plt.plot(times, abr_mean, color="b")
      plt.errorbar(times, abr_mean, yerr=abr_std / np.sqrt(abr_data.shape[0]), 
                   ecolor="0.8")
      plt.title(plot_title)
      # plt.xlim([0, 10])
      plt.ylim([-5, 5])

  