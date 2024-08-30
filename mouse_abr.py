# Code to support analysis of George's mouse ABR/ECoG recordings.  This colab
# shows how to use this code: 
# https://colab.research.google.com/drive/1wtTeslQa8BQIk9QxUfOJawU6AuhvmaDf
import csv
import dataclasses
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig
import scipy.fft as spfft
import scipy.stats as spstats
from scipy.stats import linregress

from abr import *

from typing import List, Optional, Union, Tuple

from absl import app
from absl import flags

# We save the raw data with Pickle because raw JSON doesn't support Numpy
# https://jsonpickle.github.io/
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

if os.path.exists('/content/gdrive/MyDrive/Sta nford Audiology'):
  expdir = '/content/gdrive/MyDrive/Stanford Audiology/GeorgeMouseABRData/20230823_control1_pre-0-24-1-1'


@dataclasses.dataclass
class MouseExp:
  """
  A data structure that describes one set of ABR experimental data.

  Attributes:
    filename: Where the data came from
    freq: At what frequency was the animal stimulated
    level: At what level (in dB) was the animal stimulated
    channel: Which electrode, probably 1 or 2, was recorded
    sgi: ???
    description: ??
    single_trials: ???
    paired_trials: ????
  """
  filename: str
  freq: float # Hz
  level: float # dB
  channel: int # which electrode, probably 1 or 2
  sgi: int
  description: str = ''
  single_trials: np.ndarray = None # num_trials x num_waveform_samples
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
    A MouseExp structure that describes the experiment and the single_trials
    field has the data (num_trials x num_waveform_sanples)
  """
  with open(filename, 'r', encoding='latin-1',
            newline='', errors='replace') as csvfile:
    header_names = csvfile.readline().strip().split(',')
    header_data = csvfile.readline().strip().split(',')
    header = dict(zip(header_names, header_data))

    eegreader = csv.reader(csvfile, delimiter=',')
    all_data_rows = []
    for row in eegreader:
      if len(row) > 10: # Arbitrary
        row_vals = [float(r.replace('\0', '')) for r in row if r]
        all_data_rows.append(row_vals)

  exp = MouseExp(filename=filename,
                  sgi=int(header['sgi']),
                  channel=int(header['channel']),
                  freq=float(header['Freq(Hz)']),
                  level=float(header['Level(dB)']),
                  description=header['subject'],
                  single_trials=np.array(all_data_rows)
                  )
  return exp

def read_all_mouse_dir(expdir: str, debug=False) -> List[MouseExp]:
  """
  Read in all the mouse experiments in the given directory.

  Args:
    expdir:  Where to find the experimental for this animal

  Returns:
    List of MouseExp structures.
  """
  all_exp_files = [f for f in os.listdir(expdir)
                     if os.path.isfile(os.path.join(expdir, f)) and
                     f.endswith('.csv')]

  all_exps = []
  for f in all_exp_files:
    if debug:
      print(f' Reading {f}')
    exp = read_mouse_exp(os.path.join(expdir, f))
    all_exps.append(exp)
  return all_exps


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
  good = []
  for exp in all_exps:
    if freq and freq != exp.freq:
        continue
    if level and level != exp.level:
        continue
    if channel and channel != exp.channel:
        continue
    good.append(exp)
  return good


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
    data: A matrix of shape num_trials num_samples, opposite of what the rest
      of the routines that follow need.

  Returns:
    A matrix of shape num_samples x num_trials, transposed from the original.
  """
  if remove_dc:
    data = remove_offset(data.T)  # Now data is time x num_trials
  else:
    data = data.T
  if remove_artifacts:
    data = reject_artifacts(data)
  if bandpass_filter:
    #Bidelman used 90-2000?
    data = butterworth_filter(data, lowcut=low_filter, highcut=high_filter, 
                              fs=mouse_sample_rate)
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

def calculate_dprime(data: np.ndarray,
                     noise_data: Optional[np.ndarray] = None,
                     debug=False) -> float:
  """
  Calculate the d-prime of the average response.  Form a model of the ABR signal
  by averaging all the trials together.  Cross correlate each trial with the
  model.  That forms a histogram we call H1.  Then shuffle each trial in time,
  and perform the same calculation to form the null hypothesis, H2.  Calculate
  the difference between these two empirical distributions, and normalize by
  the geometric mean of their standard deviations.

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
  dprime = (np.std(h1_response) - np.std(h2_response)) / np.sqrt(np.std(h1_response)*np.std(h2_response))
  if debug:
    counts, bins = np.histogram(h1_response, bins=40)
    plt.plot((bins[:-1]+bins[1:])/2.0, counts, label='signal_trial')
    counts, bins = np.histogram(h2_response)
    plt.plot((bins[:-1]+bins[1:])/2.0, counts, label='noise trial')
    plt.legend()
    plt.title('Histrogam of covariance of channel 2 (with and without signal)')
  return dprime


def calculate_all_dprimes(all_exps: List[MouseExp]) -> Tuple[np.ndarray,
                                                             List[float],
                                                             List[float],
                                                             List[int]]:
  """
  Calculate the d-prime for all the experiments.  Preprocess each experiment
  uisng the preprocess_mouse_data function.  The calculate the d' for each
  set of experiments with the same frequency and level.

  Args:
    all_exps: a list containing experiments in MouseExp format, before
      preprocessing.

  Returns:
    A tuple consisting of: a 3d array of dprimes, for each experiment, and
    the corresponding frequences, levels, and channels for each array dimension.
  """
  all_exp_levels = sorted(list(set([exp.level for exp in all_exps])))
  all_exp_freqs = sorted(list(set([exp.freq for exp in all_exps])))
  all_exp_channels = sorted(list(set([exp.channel for exp in all_exps])))

  dprimes = np.nan*np.zeros((len(all_exp_freqs), len(all_exp_levels),
                             len(all_exp_channels)))
  for i, freqs in enumerate(all_exp_freqs):
    for k, channel in enumerate([1, 2]):
      # Find the noisy data for this combination of frequency and channel
      noise_exp = find_noise_exp(all_exps, freq=freqs, channel=channel)
      if noise_exp is None:
        print(f'Found no noise data for freq={freqs}, channel={channel}')
        continue
      noise_data = preprocess_mouse_data(noise_exp.single_trials)

      for j, levels in enumerate(all_exp_levels):
        exps = find_exp(all_exps, freq=freqs, level=levels, channel=channel)
        if len(exps) > 1:
          #print(f'Found too many examples for freq={freqs}, level={levels}, '
          #      f'channel={channel}: {len(exps)}')
          pass
        elif len(exps) == 0:
          print(f'Found ZERO examples for freq={freqs}, level={levels}, '
                f'channel={channel}: {len(exps)}')
        signal_data = preprocess_mouse_data(exps[0].single_trials)
        dprimes[i, j, k] = calculate_dprime(signal_data, noise_data)
  return dprimes, all_exp_freqs, all_exp_levels, all_exp_channels


def plot_dprimes(dprimes: np.ndarray, all_exp_freqs: List[float],
                 all_exp_levels: List[float], all_exp_channels: List[int]):
  """Create a plot summarizing the d' collected by the calculate_all_dprimes
  routine above.  Show d' versus level, for each frequency and channel pair.

  Args:
    dprimes: a 3d array of dprimes, for all experiments, as a function of
    frequences, levels, and channels.
  """
  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color']
  plt.figure(figsize=(10, 8))

  for i, freqs in enumerate(all_exp_freqs):
    for k, channel in enumerate(all_exp_channels):
      if channel == 2:
        linestyle = '--'
      else:
        linestyle = '-'
      plt.plot(all_exp_levels, dprimes[i, :, k],
              label=f'freq={freqs}, channel={channel}',
              linestyle=linestyle,
              color=colors[i])
  plt.title('D-Prime versus Presentation Level')
  plt.legend()
  plt.xlabel('Level (dB)');


# Look for all the directories that seem to contain George's mouse data.
def find_all_mouse_data(mouse_data_dir: str) -> List[str]:
  all_exp_dirs = [x[0] for x in os.walk(mouse_data_dir)
                  if 'analyze' not in x[0] and 'bad' not in x[0] and
                    'traces' not in x[0]]
  return all_exp_dirs

# all_exp_dirs = find_all_mouse_data()


def load_exp_dir(exp_dir: str) -> List[MouseExp]:
  """
  Load all the experiments in the given directory.
  Still need to preprocess them,

  Args:
    exp_dir:

  Returns:
    A list of MouseExp structures.
  """
  pickle_file = os.path.join(exp_dir, 'mouse_exp.pkl')
  if os.path.exists(pickle_file):
    with open(pickle_file, 'r') as f:
      all_trials = jsonpickle.decode(f.read())
      print(f'  Found {len(all_trials)} experiments')
      return all_trials
  else:
    print(f'Could not find pickled data in {pickle_file}')

###############  Cache all the Mouse CSV files ###########################

mouse_data_pickle_name = 'mouse_exp.pkl'
mouse_summary_pickle_name = 'mouse_summary.pkl'
mouse_dprime_pickle_name = 'mouse_dprime.pkl'

def load_cached_mouse_data(d: str) -> List[MouseExp]:
  pickle_file = os.path.join(d, mouse_data_pickle_name)
  if os.path.exists(pickle_file):
    with open(pickle_file, 'r') as f:
        all_trials = jsonpickle.decode(f.read())
        return all_trials
  return None

def cache_mouse_data(d: str, load_data: bool = False):
  """
  Cache the CSV files in one of George's mouse recording folders.
  Check first to see if we have the cache file.
  If we have it and load_data is true return it.
  If the cache file is missing, parse all the CSV files and create
  a new cache file.

  Args:
    exp_dir: Which data directory to read and cache.
    load_data: Whether to return the cached data if it is there.

  Returns:
    A list of MouseExp structures.
  """
  pickle_file = os.path.join(d, mouse_data_pickle_name)
  if os.path.exists(pickle_file):
    if load_data:
      with open(pickle_file, 'r') as f:
        all_trials = jsonpickle.decode(f.read())
        return all_trials
    print(f'{d} exists. Skipping')
    return None
  try:
    print(f'Reading {d}')
    all_trials = read_all_mouse_dir(d, debug=True)
  except Exception as e:
    print(f'Could not read {d} because of {repr(e)}')
    print(' Skipping')
    return None
  with open(pickle_file, 'w') as f:
    f.write(jsonpickle.encode(all_trials))
    print(f' Found {len(all_trials)} experiments')
  return all_trials

def cache_mouse_summary(directory: str, all_trials):
  summaries = []
  for t in all_trials:
    t.single_trials = None
    summaries.append(t)
  pickle_file = os.path.join(directory, mouse_summary_pickle_name)
  with open(pickle_file, 'w') as f:
    f.write(jsonpickle.encode(summaries))
  return summaries

def read_mouse_summary(d):
  pickle_file = os.path.join(d, mouse_summary_pickle_name)
  if os.path.exists(pickle_file):
    with open(pickle_file, 'r') as f:
      all_trials = jsonpickle.decode(f.read())
      return all_trials
  return None

def cache_all_dirs(all_exp_dirs: List[str],
                   return_results: bool = False):
  """
  Read all of George's ABR directories, parse the CSV files, and store the lists
  of MouseExp structures in a pickle file.
  This routine walks all the directories in the input list.

  Args:
    all_exp_dirs: Where to find all the experimental data.
  """
  all_summaries = []
  for d in all_exp_dirs:
    print(f'Caching {d}')
    try:
      summary = read_mouse_summary(d)
      if summary is None:
        all_trials = cache_mouse_data(d, load_data=True)
        if all_trials is None:
          continue
        summary = cache_mouse_summary(d, all_trials)
      if return_results:
        all_summaries.append(summary)
    except Exception as e:
      print(f'Could not read {d} because of {e}')
      print(' Skipping')
      continue
  return all_summaries


def cache_all_dprimes(all_exp_dirs: List[str]) -> List:
  all_dprimes = {}

  for d in all_exp_dirs:
    try:
      print(f'Calculating dprime for {d}')
      all_exps = load_cached_mouse_data(d)
      all_dprimes[d] = calculate_all_dprimes(all_exps)
    except Exception as e:
      print(f'Could not calculate dprimes for {d} because of {e}')
  return all_dprimes

def summarize_all_dprimes(all_exp_dirs: List[str]):
  for d in all_exp_dirs:
    try:
      print(f'Summarizing data in {d}')
      all_exps = load_exp_dir(d)
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


def load_dprime_data():
  pickle_file = os.path.join(GeorgeMouseDataDir, 'all_dprimes.pkl')
  with open(pickle_file, 'r') as f:
    all_dprimes = jsonpickle.decode( f.read())
  return all_dprimes

def summarize_dprime_data(all_dprimes):
  for k in all_primes:
    dprimes, all_exp_freqs, all_exp_levels, all_exp_channels = all_dprimes[k]
    print(f'{dprimes.shape}: {k}')

def main(argv):
  all_dprimes = load_dprime_data()
  summarize_dprime_data(all_dprimes)


if __name__ == '__main__':
  app.run(main)

