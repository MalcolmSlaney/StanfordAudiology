# Code to support analysis of George's mouse ABR/ECoG recordings.  This colab
# shows how to use this code: 
# https://colab.research.google.com/drive/1wtTeslQa8BQIk9QxUfOJawU6AuhvmaDf
import csv
import dataclasses
import glob
import math
import os
import sys
import traceback

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig
import scipy.fft as spfft
import scipy.stats as spstats
from scipy.stats import linregress
from scipy.optimize import curve_fit

from absl import app
from absl import flags
from abr import *
from typing import Dict, List, Optional, Union, Tuple

# We save the raw data with Pickle because raw JSON doesn't support Numpy
# https://jsonpickle.github.io/
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()


################### d' calculations and caching ##################

@dataclasses.dataclass
class MouseExp:
  """
  A data structure that describes one set of ABR experimental data.

  Attributes:
    filename: Where the data came from
    freq: At what frequency was the animal stimulated
    level: At what level (in dB) was the animal stimulated
    channel: Which electrode, probably 1 or 2, was recorded
    sgi: Signal generation index (TDT status)
    description: ??
    single_trials: Array of waveform data for one preparation.
    paired_trials: ????
  """
  filename: str  # Full filename of the CSV mouse waveform data
  basename: str  # Just the experiment description part of the filename
  freq: float # Hz
  level: float # dB
  channel: int # which electrode, probably 1 or 2
  sgi: int # TDT Stimulus generation index (i.e. freq and level)
  description: str = ''
  single_trials: np.ndarray = None # num_waveform samples x num_trials
  paired_trials: np.ndarray = None

mouse_sample_rate = 24414 # From George's Experimental Notes
# Existing hardware filtering from 2.2-7500Hz.

def read_mouse_exp(filename: str) -> MouseExp:
  """
  Read a CSV file containing data about one ABR experiment.  Parse the header
  lines, and then read in the table of data.

  Args:
    filename:

  Returns:
    A MouseExp structure that describes the experiment. The single_trials
    field has the data (num_waveform_samples x num_trials), which is transposed
    from the CSV files.
  """
  all_data_rows = []
  with open(filename, 'r', encoding='latin-1',
            newline='', errors='replace') as csvfile:
    header_names = csvfile.readline().strip().split(',')
    header_data = csvfile.readline().strip().split(',')
    header = dict(zip(header_names, header_data))

    eegreader = csv.reader(csvfile, delimiter=',')
    for row in eegreader:
      if len(row) > 9: # Arbitrary
        row_vals = [float(r.replace('\0', '')) for r in row if r]
        all_data_rows.append(row_vals)

  exp = MouseExp(filename=filename,
                 basename=os.path.basename(filename),
                 sgi=int(header['sgi']),
                 channel=int(header['channel']),
                 freq=float(header['Freq(Hz)']),
                 level=float(header['Level(dB)']),
                 description=header['subject'],
                 single_trials=np.array(all_data_rows).T
                 )
  return exp


###############  Cache all the Mouse CSV files ###########################

mouse_waveforms_pickle_name = 'mouse_waveforms.pkl'
mouse_dprimes_pickle_name = 'mouse_dprime.pkl'


def find_all_mouse_directories(mouse_data_dir: str) -> List[str]:
  """Look for all the directories that seem to contain George's mouse data. Walk
  the directory tree starting at the given directory, looking for all 
  directories names that do not contain the following words: 
    analyze, bad and traces

  Args:
    mouse_data_dir: where to start looking for George's mouse data
  
  Returns:
    A list of file paths.
  """
  all_exp_dirs = [x[0] for x in os.walk(mouse_data_dir)
                  if 'analyze' not in x[0] and 'bad' not in x[0] and
                     'traces' not in x[0]]
  return all_exp_dirs


def waveform_cache_present(dir:str, 
                           waveform_pickle_name=mouse_waveforms_pickle_name):
  if os.path.exists(os.path.join(dir, waveform_pickle_name)):
    return True
  new_filename = waveform_pickle_name.replace('.pkl', f'00.pkl')
  return os.path.exists(os.path.join(dir, new_filename))


def save_waveform_cache(all_exps: List[MouseExp], dir: str, number: int, 
                        waveform_pickle_name=mouse_waveforms_pickle_name):
  """Save some of the MouseExp's objects into a cache file.  We store all the
  data from one directory into multiple cache files since they get to large to
  decode (with a single read).
  
  The cache file will be of the form mouse_waveformsXX.pkl, where XX is the 
  cache file number.
  """
  new_filename = waveform_pickle_name.replace('.pkl', f'{number:02d}.pkl')
  filename = os.path.join(dir, new_filename)
  with open(filename, 'w') as f:
    f.write(jsonpickle.encode(all_exps))
  print(f'Saved {len(all_exps)} MouseExp to {new_filename}.')


def load_waveform_cache(
    dir: str,
    waveform_pickle_name: str = mouse_waveforms_pickle_name) -> List[MouseExp]:
  filename = os.path.join(dir, waveform_pickle_name)
  wild_filename = filename.replace('.pkl', '*.pkl')
  filenames = glob.glob(wild_filename)
  all_exps = []
  for filename in filenames:
    with open(filename, 'rb') as f:
      new_data = jsonpickle.decode(f.read())
      if new_data:
        all_exps += new_data
        print(f'  Got {len(new_data)} MouseExp\'s from {filename}')
      else:
        print(f'  ** Found an empty Pickle file: {filename}')
  print(f'  Got a total of {len(all_exps)} MouseExp from {dir}')
  return all_exps


def summarize_all_data(all_exp_dirs: List[str], 
                       pickle_name=mouse_waveforms_pickle_name):
  for d in all_exp_dirs:
    try:
      print(f'Summarizing data in {d}')
      all_exps = load_waveform_cache(d, pickle_name)
      if not all_exps:
        print(f'  No experiments.')
      else:
        all_sizes = [str(e.single_trials.shape) for e in all_exps]
        all_sizes = ', '.join(all_sizes)
        print(f'  Sizes: {all_sizes}')
        print(f'  Channels: {sorted(list(set([e.channel for e in all_exps])))}')
        print(f'  Frequencies: {sorted(list(set([e.freq for e in all_exps])))}')
        # break
    except Exception as e:
      print(f'  Could not load mouse data for {d} because of {e}')


def cache_all_mouse_dir(expdir: str, 
                        waveform_pickle_name: str = mouse_waveforms_pickle_name,
                        max_files:int = 0, max_bytes:float = 10e9, 
                        debug:bool = False) -> None:
  """
  Read and cache the CSV mouse experiments in the given directory. Each 
  trial experiment is stored in a single CSV file.  This routine reads all the 
  csv files and converts them into numpy arrays, stored as pickle files.

  Args:
    expdir:  Where to find the experimental for this animal

  Returns:
    List of MouseExp structures.
  """
  def cache_size(all_trials: List[MouseExp]):
    return sum([exp.single_trials.nbytes for exp in all_exps])

  print(f'Cache_all_mouse_dir:', expdir, max_files, max_bytes)            
  all_exp_files = [f for f in os.listdir(expdir)
                     if os.path.isfile(os.path.join(expdir, f)) and
                     f.endswith('.csv')]

  all_exps = []
  cache_file_count = 0
  total_file_count = len(all_exp_files)
  for f in all_exp_files:
    if 'saline' in f:
      continue
    if debug:
      print(f'    Reading {f} ({len(all_exps)}/{total_file_count} '
            f'totaling {cache_size(all_exps)} bytes)')
    exp = read_mouse_exp(os.path.join(expdir, f))
    all_exps.append(exp)
    if max_files and len(all_exps) >= max_files:
      print('  Reached maximum limit of {max_files} files to process.')
      break
    if max_bytes and cache_size(all_exps) > max_bytes:
      save_waveform_cache(all_exps, expdir, cache_file_count, 
                          waveform_pickle_name=waveform_pickle_name)
      cache_file_count += 1
      all_exps = []
  if all_exps:
    save_waveform_cache(all_exps, expdir, cache_file_count, 
                        waveform_pickle_name=waveform_pickle_name)


def find_exp(all_exps: List[MouseExp],
             freq: Optional[float]=None,
             level: Optional[float] = None,
             channel: Optional[int] = None) -> List[MouseExp]:
  """
  Find particular experiments in the list of all experiments.

  Args:
    all_exps: a list containing experiments in MouseExp format
    freq: desired frequency (None means any freq)
    level: desired level (None means any level)
    channel: Recording channel (1 further away, 2 closer)

  Returns:
    A list of MouseExp's with the desired frequency and level.
  """
  good_ones = []
  for exp in all_exps:
    if freq  is not None and freq != exp.freq:
        continue
    if level is not None and level != exp.level:
        continue
    if channel  is not None and channel != exp.channel:
        continue
    good_ones.append(exp)
  return good_ones


def find_noise_exp(all_exps: List[MouseExp],
                   freq: Optional[float]=None,
                   channel: Optional[int] = None) -> MouseExp:
  """
  Find the experiment with the lowest sound level, to use as noisy data.

  Args:
    all_exps: a list of MouseExp
    freq: the desired frequency to use when selecting experiments
    channel: the desired channel to use when selecting experiments

  Returns:
    A mouse experiment generated from the lowest sound level.
  """
  exps = find_exp(all_exps, freq=freq, channel=channel)
  levels = [e.level for e in exps]
  if len(levels) == 0:
    return None
  i = np.argmin(levels)
  return exps[i]

################### d' calculations and caching ##################

# Maybe move to abr.py
def preprocess_mouse_data(data: np.ndarray,
                          remove_dc: bool = True,
                          remove_artifacts: bool = True,
                          bandpass_filter: bool = False,
                          low_filter: float = 0*200,
                          high_filter: float = 1000,
                          mouse_sample_rate: float = 24414) -> np.ndarray:
  """
  Preprocess the mouse data, removing the DC offset, rejecting artifacts, and
  applying a bandpass filter.

  Args:
    data: A matrix of shape num_samples x num_trials, opposite of what the rest
      of the routines that follow need.

  Returns:
    A matrix of shape num_samples x num_trials, transposed from the original.
  """
  if remove_dc:
    data = remove_offset(data)  # Now data is time x num_trials
  if remove_artifacts:
    data = reject_artifacts(data)
  if bandpass_filter:
    #Bidelman used 90-2000?
    data = butterworth_filter(data, lowcut=low_filter, highcut=high_filter, 
                              fs=mouse_sample_rate, order=6, axis=0)
  return data


def shuffle_data(data: np.ndarray) -> np.ndarray:
  """
  Shuffle the data in time.

  Args:
    data: A matrix of shape num_samples x num_trials

  Returns:
    A shuffled version of the data.
  """
  rng = np.random.default_rng() # Create a random number generator instance.
  # Make sure to copy the input data because shuffle rearranges its argument.
  data = data.copy()
  rng.shuffle(data, axis=0) # Shuffle in time
  return data


def exp_type_from_name(name: str) -> str:
  """Return the experiment type from a full pathname.  For example 
    20230810_control2_post60-0-30-2-1
  becomes control2_post60
  """
  return name.split('-', 1)[0].split('_', 1)[1]

def group_experiments(all_exps: List[MouseExp]) -> Dict[str, List[MouseExp]]:
  """Group all the experiments in a directory, based on the experiment type.
  The experiment type is the first part of the filename, after the date.
  Args:
    all_exps: A list of all the MouseExps (in a directory)
    
  Returns:
    A dictionary of groups of MouseExps.  The key is the experiment type and the
    value is a list of MouseExps.
  """
  types = set([exp_type_from_name(exp.basename) for exp in all_exps])
  exp_groups = {}
  for exp_type in types:
    this_exps = [exp for exp in all_exps 
                 if exp_type_from_name(exp.basename) == exp_type]
    exp_groups[exp_type] = this_exps
  return exp_groups


###############  Compute all the d-primes for our data #######################
@dataclasses.dataclass
class DPrimeResult(object):
  """Consolidate all the d' results for one preparation, across frequency.
  level and channel."""
  cov_dprimes: np.ndarray # 3d array indexed by frequency, level and channel
  rmses: np.ndarray # Correspomding RMS values for each group
  rms_dprimes: np.ndarray # Corresponding d' for the RMS calculations.
  freqs: List[float]
  levels: List[float]
  channels: List[int]
  spl_threshold: Optional[np.ndarray] = None
  smooth_dprimes: Optional[np.ndarray] = None


def calculate_rms(data: np.ndarray):
  """Calculate the RMS power for a set of waveform measurements.
  Note, because of the root, this is now in the amplitude domain.  Average
  over all time, returning the RMS for each trial.

  Args:
    data: an array of num_samples x num_trials
  Returns:
    An array of length num_trials.
  """
  return np.sqrt(np.mean(data**2, axis=0))


def calculate_cov_dprime(data: np.ndarray,
                         noise_data: Optional[np.ndarray] = None,
                         debug=False) -> float:
  """
  Calculate the d-prime of the covariance response.  Form a model of the ABR 
  signal by averaging all the trials together.  Cross correlate each trial with 
  the model.  That forms a histogram we call H1.  Then shuffle each trial in 
  time, and perform the same calculation to form the null hypothesis, H2.  
  Calculate the difference between these two empirical distributions, and 
  normalize by the geometric mean of their standard deviations.

  Args:
    data: A matrix of shape num_samples x num_trials
    moise_data: A matrix like data, but with only noise, no sigal.
    debug: Whte
  Returns:
    A scalar representing the d-prime.
  """
  if noise_data is None:
    noise_data = data
  shuffled_data = shuffle_data(noise_data)
  model = np.mean(data, axis=1, keepdims=True)
  h1 = model * data
  h1_response = np.sum(h1, axis=0) # Sum response over time
  h2 = model * shuffled_data
  h2_response = np.sum(h2, axis=0) # Sum response over time
  dprime = (np.mean(h1_response) - np.mean(h2_response)) / np.sqrt(np.std(h1_response)*np.std(h2_response))
  if debug:
    range = (min(np.min(h1_response), np.min(h2_response)),
             max(np.max(h1_response), np.max(h2_response)))
    counts, bins = np.histogram(h1_response, bins=40, range=range)
    plt.plot((bins[:-1]+bins[1:])/2.0, counts, label='signal_trial')
    counts, bins = np.histogram(h2_response, bins=40, range=range)
    plt.plot((bins[:-1]+bins[1:])/2.0, counts, label='noise trial')
    plt.legend()
    plt.title('Histogram of covariance')
    a = plt.axis()
    plt.text(a[0], a[2], 
             f' H1: {np.mean(h1_response):4.3G} +/- {np.std(h1_response):4.3G}\n'
             f' H2: {np.mean(h2_response):4.3G} +/-{np.std(h2_response):4.3G}\n'
             f' d\'={dprime:4.3G}\n\n\n')
  return dprime


def calculate_rmses(signal_data, noise_data, debug):
  noise_rms = calculate_rms(noise_data)
  signal_rms = calculate_rms(signal_data)
  rms = np.sqrt(np.mean(signal_rms**2))
  dprime = (np.mean(signal_rms) - np.mean(noise_rms)) / np.sqrt(np.std(signal_rms)*np.std(noise_rms))

  if debug:
    range = (min(np.min(signal_rms), np.min(noise_rms)),
            max(np.max(signal_rms), np.max(noise_rms)))
    counts, bins = np.histogram(signal_rms, bins=40, range=range)
    plt.plot((bins[:-1]+bins[1:])/2.0, counts, label='signal_trial')
    counts, bins = np.histogram(noise_rms, bins=40, range=range)
    plt.plot((bins[:-1]+bins[1:])/2.0, counts, label='noise trial')
    plt.legend()
    plt.title('Histogram of covariance')
    a = plt.axis()
    plt.text(a[0], a[2], 
            f' H1: {np.mean(signal_rms):4.3G} +/- {np.std(signal_rms):4.3G}\n'
            f' H2: {np.mean(noise_rms):4.3G} +/-{np.std(noise_rms):4.3G}\n'
            f' d\'={dprime:4.3G}\n\n\n')
          
  return rms, dprime


def calculate_all_summaries(all_exps: List[MouseExp]) -> Dict[str, 
                                                              DPrimeResult]:
  """Calculate the waveform summaries for each type of experiment within this 
  list of results.  Each result is for one experiment, at one frequency, level 
  and channel. This code groups the experiments together that share the same 
  type, based on the second component of the file name, and then computes the
  d' as a function of frequency, level and channel.

  Arg:
    all_exps: A list of MouseExps, containing the raw waveform data for each
      condition

  Returns:
    A dictionary, keyed by experiment type, containing the dprime result.
  """ 
  all_groups = group_experiments(all_exps)
  all_dprimes = {}
  for t, exps in all_groups.items():
    result = DPrimeResult(*calculate_waveform_summaries(exps))
    all_dprimes[t] = result
  return all_dprimes


def calculate_waveform_summaries(all_exps: List[MouseExp],
                                 debug_cov_not_rms: bool = True,
                                 debug_freq: Optional[float] = None,
                                 debug_levels: List[float] = [],
                                 debug_channel: Optional[int] = None,
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                           List[float],
                                           List[float],
                                           List[int]]:
  """
  Calculate the covariance and RMS d' for all trial preparations.  First gather 
  the waveforms by frequency, level and channel.  And then for each preparation
  preprocess the waveforms.  The calcuate the covariance and its d', and then
  do the same thing for RMS.

  Args:
    all_exps: a list containing experiments in MouseExp format, before
      preprocessing.
    debug_cov_not_rms: If true plot the covariance results, otherwise RMS
    debug_freq: Which frequency from this preparation to plot
    debug_levels: Which levels from this preparation to plot (a list)
    debug_channel: Which channel to plot

  Returns:
    A tuple consisting of: 
      a 3d array of d' for the covariance measure, for each experiment,
      a 3d array of RMS values,
      a 3d array of d' for the RMS measure
      Then lists of the found the corresponding frequences, levels, and channels
    The arrays are 3d and are indexed by frequency, level, and channel.
  """
  all_exp_levels = sorted(list(set([exp.level for exp in all_exps])))
  all_exp_freqs = sorted(list(set([exp.freq for exp in all_exps])))
  all_exp_channels = sorted(list(set([exp.channel for exp in all_exps])))

  plot_num = 1
  cov_dprimes = np.nan*np.zeros((len(all_exp_freqs), len(all_exp_levels),
                                len(all_exp_channels)))
  rmses = np.nan*np.zeros((len(all_exp_freqs), len(all_exp_levels),
                          len(all_exp_channels)))
  rms_dprimes = np.nan*np.zeros((len(all_exp_freqs), len(all_exp_levels),
                                len(all_exp_channels)))
  # Now loop through all the frequencies, channels, and levels.
  all_processed = 0
  all_multiprocessed = 0
  for i, freq in enumerate(all_exp_freqs):
    for k, channel in enumerate([1, 2]):
      # Find the noisy data for this combination of frequency and channel
      noise_exp = find_noise_exp(all_exps, freq=freq, channel=channel)
      if noise_exp is None:
        print(f'Found no noise data for freq={freq}, channel={channel}')
        continue
      
      noise_data = preprocess_mouse_data(noise_exp.single_trials)
      noise_rms = calculate_rms(noise_data)

      for j, level in enumerate(all_exp_levels):
        exps = find_exp(all_exps, freq=freq, level=level, channel=channel)
        if len(exps) == 0:
          print(f' Found ZERO examples for freq={freq}, level={level}, '
                f'channel={channel}: {len(exps)}')
          continue
        elif len(exps) > 1:
          print(f'  Processing {len(exps)} segments for the same preparation.')
          all_multiprocessed += 1
        all_data = []
        for exp in exps:
          all_processed += 1
          all_data.append(preprocess_mouse_data(exp.single_trials))
        signal_data = np.concatenate(all_data, axis=1)

        debug = (channel == debug_channel and freq == debug_freq and 
                 level in debug_levels)
        if debug:
          plt.subplot(2, 2, plot_num)
          plot_num += 1
        cov_dprimes[i, j, k] = calculate_cov_dprime(signal_data, noise_data, 
                                                    debug and debug_cov_not_rms)
        rms, dprime = calculate_rmses(signal_data, noise_data, 
                                      debug and not debug_cov_not_rms)
  
        signal_rms = calculate_rms(signal_data)
        rmses[i, j, k] = rms
        rms_dprimes[i, j, k] = dprime
        if debug:
          plt.title(f'freq={int(freq)}, level={int(level)}, channel={int(channel)}')
  print(f'  Processed {all_processed} CSV files, {all_multiprocessed} part of a group.')
  return cov_dprimes, rmses, rms_dprimes, all_exp_freqs, all_exp_levels, all_exp_channels


def filter_dprime_results(all_dprimes: Dict[str, DPrimeResult],
                          keep_list: List[str] = [],
                          drop_list: List[str] = [],
                          min_abr_thresh: float = 0.0,
                          max_abr_thresh: float = 1e9,
                          min_ecog_thresh: float = 0.0,
                          max_ecog_thresh: float = 1e9,
                          ) -> Tuple[List[float], 
                                     List[float]]:
  """Filter a DPrime dictionary, looking for good preparations and dropping the
  bad ones.  And setting limits on the calculate thresholds, looking for good
  and bad data.
  
  Args:
    all_dprimes: A dictionary pointing to dprime results
    keep_list: A list of strings with words from the date_preparation_name keys
      that we want to keep
    drop_list: Like keep list, but overrules with keys to drop
    min_abr_thresh: Keep preps where all thresholds are *above* this limit
    max_abr_thresh: Keep preps where all thresholds are *below* this limit
    min_ecog_thresh: Like min_abr_thresh, but using ECoG data
    max_ecog_thresh: Like max_abr_thresh, but using ECoG data
    """
  filtered_dprimes = {}
  for k in all_dprimes.keys():
    if any([l in k for l in drop_list]):
      continue
    if len(keep_list) and not any([l in k for l in keep_list]):
      continue
    
    dp = all_dprimes[k]
    if dp.spl_threshold is None:
      continue
    if not isinstance(dp.spl_threshold, np.ndarray) or dp.spl_threshold.ndim < 2:
      continue

    if np.min(dp.spl_threshold[:, 0]) < min_abr_thresh:
      continue
    if np.max(dp.spl_threshold[:, 0]) > max_abr_thresh:
      continue
    if np.min(dp.spl_threshold[:, 1]) < min_ecog_thresh:
      continue
    if np.max(dp.spl_threshold[:, 1]) > max_ecog_thresh:
      continue

    filtered_dprimes[k] = dp
  return filtered_dprimes


def plot_dprimes(dp: DPrimeResult):
  """Create a plot summarizing the d' of the covariance data collected by the 
  calculate_all_summaries routine above.  Show d' versus level, for each 
  frequency and channel pair.

  Args:
    dprimes: a 3d array of dprimes, for all experiments, as a function of
    frequences, levels, and channels.
  """
  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color']
  plt.figure(figsize=(10, 8))

  for i, freqs in enumerate(dp.freqs):
    for k, channel in enumerate(dp.channels):
      if channel == 2:
        linestyle = '--'
      else:
        linestyle = '-'
      plt.plot(dp.levels, dp.cov_dprimes[i, :, k],
              label=f'freq={freqs}, channel={channel}',
              linestyle=linestyle,
              color=colors[i])
  plt.title('D-Prime versus Presentation Level')
  plt.legend()
  plt.xlabel('Level (dB)');


def cache_dprime_data(d: str, 
                      dprimes: Dict[str, DPrimeResult],
                      dprime_pickle_name: str):
  """
  Cache all the dprime data in one of George's mouse recording folders.
  If we don't have the cache file, calculate the dprime data
  a new cache file.  If return_data is true, return the dictionary of waveform
  data, reading it back in if we didn't compute it here.

  Args:
    exp_dir: Which data directory to read and cache.
    waveform_pickle_name: The name of the waveform cache file.
    load_data: Whether to return the cached data if it is there.

  Returns:
    A list of MouseExp structures.
  """
  pickle_file = os.path.join(d, dprime_pickle_name)
  with open(pickle_file, 'w') as f:
    f.write(jsonpickle.encode(dprimes))
    print(f'  Cached data for {len(dprimes)} types of dprime experiments.')

###############  Summarize and smooth the d' data ##############################


def get_all_dprime_data(
    dirs, 
    pickle_name: str = mouse_dprimes_pickle_name) -> Dict[str, DPrimeResult]:
  all_dprimes = {}
  for d in dirs:
    animal_date = os.path.basename(d)
    pickle_file = os.path.join(d, pickle_name)
    if os.path.exists(pickle_file):
      with open(pickle_file, 'rb') as fp:
        mouse_dprime = jsonpickle.decode(fp.read())
      mouse_dprime2 = {}
      for k in mouse_dprime.keys():
        mouse_dprime2[f'{animal_date}_{k}'] = mouse_dprime[k]
      print(f'Added {len(mouse_dprime2)} d\' results added from {pickle_file}')
      all_dprimes.update(mouse_dprime2)
  return all_dprimes


class BilinearInterpolation(object):
  def __init__(self):
    self._xdata = []
    self._ydata = []

  def fit(self, xdata, ydata):
    i = np.argsort(xdata)
    if len(xdata) != len(ydata):
      raise ValueError('Unequal array sizes passed to fit')
    self._xdata = np.asarray(xdata)[i]
    self._ydata = np.asarray(ydata)[i]

  def eval(self, x):
    if isinstance(x, list) or (isinstance(x, np.ndarray) and x.size > 1):
      return [self.eval(f) for f in x]
    if len(self._xdata) < 2:  # Not enough data for interpolation
      return self._ydata
    if x <= self._xdata[0]: 
      i = 0
    elif x >= self._xdata[-2]:
      i = len(self._xdata)-2
    else:
      i = np.nonzero(x > self._xdata)[0][-1]
    delta = (x - self._xdata[i])/(self._xdata[i+1]-self._xdata[i])
    return self._ydata[i]*(1-delta) + self._ydata[i+1]*delta

  def threshold(self, y):
    if len(self._xdata) < 2:
      return self._xdata[0]
    if y <= self._ydata[0]:
      i = 0
    elif y >= self._ydata[-2]:
      i = len(self._ydata)-2
    else:
      i = np.nonzero(y > self._ydata)[0][-1]
    delta = (y - self._ydata[i])/(self._ydata[i+1]-self._ydata[i])
    return self._xdata[i]*(1-delta) + self._xdata[i+1]*delta
  
class PositivePolynomial(object):
  """A class that lets us fit a quadratic function with all positive
  coefficients to some d' data.
  """
  def __init__(self):
    self._a = 0
    self._b = 0
    self._c = 0

  def quadratic(_, x, a, b, c):
    return a + b*x + c*x**2 

  def fit(self, xdata, ydata, bounds=([0, 0, 0], [np.inf, np.inf, np.inf])):
    if len(xdata) != len(ydata):
      raise ValueError('Unequal array sizes passed to fit')
    (self._a, self._b, self._c), _ = curve_fit(self.quadratic, xdata, ydata, 
                                               bounds=bounds)
  
  def eval(self, x):
    return self.quadratic(x, self._a, self._b, self._c)

  def threshold(self, threshold, debug=False):
    """Find the level when the quadratic function passes the threshold."""
    a = self._a - threshold
    b = self._b
    c = self._c
    if debug:
      xdata = np.linspace(-4, 4, 100)
      plt.plot(xdata, a + b*xdata + c*xdata**2)
      plt.axhline(0, ls='--')
      
    roots = [(-b + math.sqrt(b**2-4*a*c))/(2*c),
             (-b - math.sqrt(b**2-4*a*c))/(2*c)]
     # Filter for positive roots and select minimum
    positive_roots = [r for r in roots if r > 0]
    
    if positive_roots:
        root = np.min(positive_roots)
        return root  # Return the single root value
    return np.nan  # Or any other appropriate value for no positive roots


def add_threshold(dprimes_result: DPrimeResult, dp_criteria=2, 
                  fit_method: str = 'bilinear',
                  plot=False) -> None:
  """Add the SPL threshold to a DPrimeResult.  This is done by fitting a
  either a polynomial or bilinear model to the d' data at each frequency and 
  channel, and then using this smooth model to predict the level where d' 
  exceeds the desired level.
  Args:
    dprimes_result: Consolidated estimate of the d' for each frequency, level
      and channel
    dp_criteria: Desired d' threshold
    fit_method: Either 'bilinear' or 'polynomial'
    plot: Generate plot showing curve fits
  Returns:
    Nothing.  dprimes_result is modified in place, adding:
      1) the spl_threshold where d' gets above the dp_criteria, and 
      2) smoothed_dprimes, which gives the d' estimate, after polynomial 
         smoothing at the same frequencies, levels, and channels as the incoming
         dprime array.
  """
  if len(dprimes_result.channels) != 2:
    return np.zeros((0, 2))
  spl_threshold = np.zeros((len(dprimes_result.freqs),
                            len(dprimes_result.channels)))
  min_level = min(dprimes_result.levels)
  max_level = max(dprimes_result.levels)
  plot_levels = np.linspace(min_level, max_level, 100)
  channel_names = ['', 'ABR', 'ECoG']

  color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

  smoothed = np.zeros((len(dprimes_result.freqs),
                       len(dprimes_result.levels),
                       len(dprimes_result.channels)))
  dprimes = dprimes_result.cov_dprimes
  for i, freq in enumerate(dprimes_result.freqs):
    for j, channel in enumerate(dprimes_result.channels):
      levels = dprimes_result.levels
      dprimes = dprimes_result.cov_dprimes[i, :, j]
      cp = None
      try:
        if fit_method == 'bilinear':
          interp = BilinearInterpolation()
        elif fit_method == 'polynomial':
          interp = PositivePolynomial()
        else:
          assert ValueError(f'Unknown fit method: {fit_method}')
        interp.fit(levels, dprimes)
        r = interp.threshold(dp_criteria)
        # Check if r is a list and take the first element if it is
        # This ensures we store a single numeric value in db_at_threshold
        if isinstance(r, list):
            r = r[0] if r else np.nan
      except Exception as error:
        print(f'Could not fit levels: {error}')
        print(traceback.format_exc())
        r = np.nan
      spl_threshold[i, j] = r
      smoothed[i, :, j] = interp.eval(np.asarray(dprimes_result.levels))
      if interp and plot:
        if channel == 1:
          ls = '--'
        else:
          ls = '-'
        plt.plot(plot_levels, [interp.eval(l) for l in plot_levels], 
                label=f'{channel_names[channel]} at {freq}Hz',
                color=color_list[i], ls=ls)
        plt.plot(levels, dprimes, 'x', 
                color=color_list[i])
        plt.axhline(dp_criteria, color='r', ls=':')
        plt.axvline(r, color='r', ls=':')
  if plot:
    plt.legend()
    plt.xlabel('Sound Level (dB)')
    plt.ylabel('d\'')
  dprimes_result.spl_threshold = spl_threshold
  dprimes_result.smooth_dprimes = smoothed


def add_all_thresholds(all_dprimes: Dict[str, DPrimeResult], dp_criteria=2,
                       fit_method='bilinear'):
  """Process all the dprime structures we have, using the add_threshold function.
  """
  for k in all_dprimes.keys():
    add_threshold(all_dprimes[k], dp_criteria=dp_criteria, 
                  fit_method=fit_method)


def accumulate_thresholds(all_dprimes: Dict[str, DPrimeResult],
                          freqs: Optional[List[float]] = None,
                          max_spl=120) -> Tuple[List[float], 
                                                List[float], float]:
  """Accumulate all the thresholds for ABR and ECoG data, across all the data
  we have.  Filter out the preparation names we do and don't want.  And remove
  remove any preparations where the computed threshold is greater than max_spl,
  indicating that we got no data for this prep.

  Args:
    all_dprimes: Dictionary keyed by the date_prep_name and pointing to the 
      d' data for this preparation.  In particular we look at the spl_threshold,
      which should already be calculated by add_threshold() and the 
    freqs: List of frequencies to keep. The default is to return the d' 
      regardless of test frequency.
    max_spl: Filter results using this threshold before computing the Pearson
      correlation.
  Returns:
    1) List of ABR thresholds
    2) List of corresponding ECoG thresholds
    3) the Pearson correlation between the ABR and ECoG thresholds
  """
  all_abr = []
  all_ecog = []
  for k in all_dprimes.keys():
    dp = all_dprimes[k]
    if dp.spl_threshold is None:
      continue
    if freqs is not None:
      if not isinstance(freqs, list):
        freqs = [freqs,]
      freq_indices = [dp.freqs.index(f) for f in freqs if f in dp.freqs]
    else:
      freq_indices = range(len(dp.freqs))
    # print(freq_indices, dp.spl_threshold.shape)
    all_abr.append(dp.spl_threshold[freq_indices, 0].flatten())
    all_ecog.append(dp.spl_threshold[freq_indices, 1].flatten())
  if len(all_abr) == 0:
    print(f'Found no data for frequency list: {",".join(freqs)}')
    return [], [], 0.0
  all_abr = np.concatenate(all_abr)
  all_ecog = np.concatenate(all_ecog)

  # Calculate the Pearson correlation using hte "good" data.
  good = np.logical_and(np.logical_and(np.isfinite(all_abr), all_abr < max_spl),
                        np.logical_and(np.isfinite(all_ecog), all_ecog < max_spl))
  abr_thresh = all_abr[good]
  ecog_thresh = all_ecog[good]
  pearson_r = spstats.pearsonr(abr_thresh, ecog_thresh).statistic

  return all_abr, all_ecog, pearson_r


def plot_threshold_scatter(abr_thresh: np.ndarray, ecog_thresh: np.ndarray, 
                           title:str = 'Comparson of Threshold', 
                           axis_limit: float = 120,
                           color:str = 'b', draw_unit:bool = True):
  """Plot a comparison of ABR and ECoG thresholds.
  Args:
    abr_thresh: A 1d array of ABR thresholds for different preparations.
    ecog_thresh: A array of ECoG thresholds corresponding to abr_thresh
    title: The title to place on the graph.
    axis_limit: The maximum x and y limits on the axis, but no bigger than the
      actual data.
    color: Which plot color to use for this data
    draw_unit: Draw the 1:1 line for comparison.
  """
  good = np.logical_and(np.isfinite(abr_thresh), np.isfinite(ecog_thresh))
  abr_thresh = abr_thresh[good]
  ecog_thresh = ecog_thresh[good]

  plt.plot(abr_thresh, ecog_thresh, 'x', color=color)
  plt.xlabel('ABR (Channel 1) Threshold (dB)')
  plt.ylabel('ECoG (Channel 2) Threshold (dB)')
  if title:
    plt.title(title)
  axis_limit = min(min(np.max(abr_thresh), np.max(ecog_thresh)), axis_limit)
  plt.xlim(0, axis_limit)
  plt.ylim(0, axis_limit)
  if draw_unit:
    a=max(plt.axis())
    plt.plot([0, a], [0, a], '--');


def find_dprime(all_dprimes: Dict[str, DPrimeResult], 
                spl: float) -> Tuple[np.ndarray, np.ndarray]:
  """Find the d' for all data in this dictionary of preparations at one SPL. 
  Use the smooth d' data, which is best calculated with bilinear interpolaton.

  Args:
    all_dprimes: Dictionary mapping preparation name to d' results.
    spl: Which sound pressure level (SPL) to query the d'.  Must be one of the
      measured SPLs, as computed by the smooth polynomial.
  Returns:
    Matched ABR and ECoG arrays with the d' at the given SPL.
  """
  abr_90s = []
  ecog_90s = []
  for k in all_dprimes:
    dp = all_dprimes[k]
    if dp.smooth_dprimes is None or spl not in dp.levels or dp.smooth_dprimes.shape[2] < 2:
      continue
    index = dp.levels.index(spl)
    abr_90s.append(dp.smooth_dprimes[:, index, 0])
    ecog_90s.append(dp.smooth_dprimes[:, index, 1])

  return np.concatenate(abr_90s), np.concatenate(ecog_90s)
  
###############  Main program, so we can run this offline ######################

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'waveforms', ('waveforms', 'dprimes', 'check'),
                  'Which processing to do on this basedir.')
flags.DEFINE_string('basedir',
                    'drive/Shareddrives/StanfordAudiology/GeorgeMouseABR/CAP_ABR',
                    'Base directory to find the ABRPresto mouse data')
flags.DEFINE_string('waveforms_cache', mouse_waveforms_pickle_name,
                    'Where to cache all the waveforms in this directory')
flags.DEFINE_string('dprimes_cache', mouse_dprimes_pickle_name,
                    'Where to cache the dprimes in this directory')
flags.DEFINE_string('filter', '', 'Which directories to process, ignore rest.')
flags.DEFINE_integer('max_cache_gbytes', 10, 
                     'Maximum size of one cache file (GBytes).')


def waveform_caches_present(dir:str, waveform_pickle_name:str) -> int:
  """Check to make sure that all the waveform cache files in this directory
  are not emoty and there is at least one good one.
  """
  filepath = os.path.join(dir, waveform_pickle_name)
  wild_filename = filepath.replace('.pkl', '*.pkl')
  filenames = glob.glob(wild_filename)
  good_files = 0
  for filename in filenames:
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
      good_files += 1
    else:
      print(f'Found a zero length waveform cache file: {filename}')
  return good_files
 

def cache_waveform_one_dir(dir:str, waveform_pickle_name:str, 
                           max_files:int = 0, max_bytes:float = 10e9):
  """Read all the CSV files and convert them into pickled numpy arrays.  CSV
  files take a long time to read and parse, so this is an important speedup.
  """
  num_good = waveform_caches_present(dir, waveform_pickle_name)
  if num_good:
    print(f'Skipping waveforms and dprimes in {dir} because they are '
          f'{num_good} cached files.')
    return
  print(f'Processing CSV waveforms in {dir}')
  all_exps = cache_all_mouse_dir(dir, waveform_pickle_name, debug=True,
                                 max_files=max_files, max_bytes=max_bytes)


def cache_dprime_one_dir(dir:str, 
                         waveform_cache_name:str, dprime_cache_name:str):
  all_exps = load_waveform_cache(dir, waveform_cache_name)
  if all_exps:
    dprimes = calculate_all_summaries(all_exps)
    cache_dprime_data(dir, dprimes, dprime_cache_name)
  else:
    print(f'  No waveform data to process for dprimes.')

def main(_):
  if FLAGS.mode == 'waveforms':
    all_mouse_dirs = find_all_mouse_directories(FLAGS.basedir)
    for dir in all_mouse_dirs:
      if FLAGS.filter in dir:
        # waveform_cache = os.path.join(dir, FLAGS.waveforms_cache)
        # dprime_cache = os.path.join(dir, FLAGS.dprimes_cache)
        cache_waveform_one_dir(dir, FLAGS.waveforms_cache, 
                        max_bytes=FLAGS.max_cache_gbytes*1e9)
  elif FLAGS.mode == 'dprimes':
    all_mouse_dirs = find_all_mouse_directories(FLAGS.basedir)
    for dir in all_mouse_dirs:
      if FLAGS.filter in dir:
        cache_dprime_one_dir(dir, FLAGS.waveforms_cache, FLAGS.dprimes_cache)
  else:
    print(f'Unknown processing mode: {FLAGS.mode}')

if __name__ == '__main__':
  app.run(main)

