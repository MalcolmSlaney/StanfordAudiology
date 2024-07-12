from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union
from scipy.signal import butter, freqz, lfilter, sosfilt, sosfiltfilt
from sklearn.linear_model import Ridge, LinearRegression
from urllib.request import DataHandler

""" 
To use this code in Colab do the following:
  !git clone https://github.com/MalcolmSlaney/StanfordAudiology.git
  !ls StanfordAudiology
  import sys
  sys.path.insert(0,'StanfordAudiology')

  # Note this syntax (import *) is in general a bad idea.  But it's a good 
  # solution in colab so you can easily redefine a function that was already 
  # defined in the abr.py file.  Just don't forget to check in the new version 
  # at some point.
  #   https://www.geeksforgeeks.org/why-import-star-in-python-is-a-bad-idea/

  from StanfordAudiology.abr import *

You might also want to do
  from google.colab import drive
  drive.mount('/content/drive')
"""

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


def design_butterworth_filter(lowcut: float, highcut: float, 
                            fs: float, order: int=6) -> np.ndarray:
  """Design a butterworth filter with the indicated parameters. Returns the filter 
  coefficients (for each second order section) as an array.
  """
  if lowcut <= 0:
    filter_type = 'lowpass'
    freqs = highcut
  elif highcut >= fs/2:
    filter_type = 'highpass'
    freqs = lowcut
  else:
    filter_type = 'bandpass'
    freqs = [lowcut, highcut]
  # print(f'Designing a {filter_type} filter with critical '
  #       f'frequencies {freqs}.')
  return butter(order, freqs, fs=fs, btype=filter_type, output='sos')

def butterworth_filter(data: np.ndarray, lowcut: float, highcut: float, 
                       fs: float, order: int=5, axis=-1) -> np.ndarray:
  """FIlter an array of signals with a Butterworth (smooth passband) 
  filter with the given low-frequency and high-frequency cutoffs.  All 
  frequencies are in Hz.

  This function filters the data twice, once forward and once backward.  This
  means the effective order is twice the requested order, but more importantly
  it cancels the filter's phase.

  Args
    data: A one-dimensional signal to be filtered
    lowcut: The low frequency cutoff for the filter.  If it is zero, then a
      low-pass filter is designed (instead of a bandpass.)
    highcut: The hiqh frequency cutoff for the filter.  If it is greater 
      than or equal to the Nyquist frequency (fs/2) then a low-pass filter
      is computed.
    fs: Sampling rate for the data (Hz, like the cutoffs)
    order: The polynomial order in the filter implementation.  Higher orders
      lead to sharper transitions, and more computational time. 
    axis: Which axis of the data over which to filter.

  Returns:
    The original data filtered as requested.        
  """
  sos_sections = design_butterworth_filter(lowcut, highcut, fs, order=order)
  y = sosfiltfilt(sos_sections, data, axis=axis)
  return y


def remove_offset(abr_data: np.ndarray) -> np.ndarray:
  """Remove the mean from each channel. This is good to do this way (instead
  of using the bandpass filter, because there is no startup energy due to a 
  non-zero offset)
  
  Args:
    abr_data: The raw data of shape num_samples x num_channels.
    
  Returns:
    The same shape data, with the mean in each channel removed.
  """
  # Important to use 64 bit arithmetic otherwise "Catastrophic cancellation" 
  # causes imprecision, especially on long signals.
  filtered_data = abr_data - np.mean(abr_data.astype(np.float64), 
                                     axis=0, keepdims=True)
  assert filtered_data.shape[0] == abr_data.shape[0]
  return filtered_data


def rereference(eeg: np.ndarray, channel: Optional[int]) -> np.ndarray:
  """Re reference the EEG data, by using a new *ground* signal.  This reference
  can either be a single channel (the ground) or the mean of all the channels.
  
  Args:
    eeg: The EEG data, of size num_samples x num_channels
    channel: Either an integer channel number (zero based) or None to indicate
      a the mean of all channels is the reference.
      
  Returns:
    The modified array, with the new ground reference.
  """
  if isinstance(channel, int):
    reference = eeg[:, channel:channel+1]
  else:
    reference = np.mean(eeg, keepdims=True, axis=1)
  return eeg - reference

def extract_epochs(data: np.ndarray,
                   locs: Union[int, List[int], np.ndarray],
                   length: int) -> np.ndarray:
  """
  Extract the click response from a multi-channel EEG recording.
  
  Args:
    data: The EEG data, of size num_samples (time) x num_channels
    locs: Where each click occurs.  Either an integer indicating a periodic
      stimuli, or a list of integers indicating the time of each click.
    length:  How many samples to extract after each click.

  Returns:
    A tensor of shape length (time) x num_epochs x num_channels
  """
  num_samples, num_channels = data.shape
  if isinstance(locs, int):
    locs = list(range(0, num_samples, locs))
  num_epochs = len(locs)
  epochs = np.zeros((length, num_epochs, num_channels), np.float32)
  for i, sample_start in enumerate(locs):
    if sample_start+length <= num_samples:
      epochs[:, i, :] = data[sample_start:sample_start+length, :]
  return epochs


def epoch_bipolar_data(clean_eeg: np.ndarray, 
                       positive_indices: Union[List[int], np.ndarray], 
                       negative_indices: Union[List[int], np.ndarray], 
                       epoch_length) -> np.ndarray:
  """
  Accumulate positive and negative click stimuli, and flip the negative
  stimuli to get an average response. This was originally done so that
  we could cancel the electrical noise, but that's not necessary if the stimuli
  are flipped, and the brain will respond the same way either way.  Thus 
  cancelling the electrical noise (but not the brain noise.)

  Args:
    clean_eeg: The EEG data, of size num_samples x num_channels
    positive_indices: The location of the postitive clicks (sanple #).
    negative_indices: The location of the negative clicks (sanple #).
    epoch_length: How much data to extract after each click.

  Returns:
    A tensor of shape length (time) x num_epochs x num_channels
  """
  positive_epochs = extract_epochs(data=clean_eeg, locs=positive_indices,
                                   length=epoch_length)
  negative_epochs = extract_epochs(data=clean_eeg, locs=negative_indices,
                                   length=epoch_length)
  epoch_count = min(positive_epochs.shape[1], negative_epochs.shape[1])
  epochs = (positive_epochs[:, :epoch_count, :] -
            negative_epochs[:, :epoch_count, :])
  return epochs


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

  