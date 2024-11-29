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
from scipy.optimize import curve_fit


from abr import *

from typing import Dict, List, Optional, Union, Tuple

from absl import app
from absl import flags

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
    sgi: ???
    description: ??
    single_trials: ???
    paired_trials: ????
  """
  filename: str  # Full filename of the CSV mouse waveform data
  basename: str  # Just the experiment description part of the filename
  freq: float # Hz
  level: float # dB
  channel: int # which electrode, probably 1 or 2
  sgi: int # TDT Stimulus generation index (i.e. freq and level)
  description: str = ''
  single_trials: np.ndarray = None # num_waveforms x num_trials
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

def read_all_mouse_dir(expdir: str, debug=False) -> List[MouseExp]:
  """
  Read in all the mouse experiments in the given directory. Each experiment
  is stored in a single CSV file.  This routine reads all the csv files and 
  returns a list of MouseExp's

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
    if 'saline' in f:
      continue
    if debug:
      print(f'    Reading {f}')
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
  dprime = (np.mean(h1_response) - np.mean(h2_response)) / np.sqrt(np.std(h1_response)*np.std(h2_response))
  if debug:
    counts, bins = np.histogram(h1_response, bins=40)
    plt.plot((bins[:-1]+bins[1:])/2.0, counts, label='signal_trial')
    counts, bins = np.histogram(h2_response)
    plt.plot((bins[:-1]+bins[1:])/2.0, counts, label='noise trial')
    plt.legend()
    plt.title('Histogram of covariance of channel 2')
    a = plt.axis()
    plt.text(a[0], a[2], 
             f' H1: {np.mean(h1_response):4.3G} +/- {np.std(h1_response):4.3G}\n'
             f' H2: {np.mean(h2_response):4.3G} +/-{np.std(h2_response):4.3G}\n'
             f' d\'={dprime:4.3G}\n\n\n')
  return dprime

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
  dprimes: np.ndarray # 3d array indexed by frequency, level and channel
  freqs: List[float]
  levels: List[float]
  channels: List[int]
  spl_threshold: Optional[np.ndarray] = None # db SPL for frequency by channel
  smooth_dprimes: Optional[np.ndarray] = None # Like dprimes, but from poly fit


def calculate_all_dprimes(all_exps: List[MouseExp]) -> Dict[str, DPrimeResult]:
  """Calculate the dprime for each type of experiment within this list of 
  results.  Each result is for one experiment, at one frequency, level and 
  channel. This code groups the experiments together that share the same type,
  based on the second component of the file name, and then computes the
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
    result = DPrimeResult(*calculate_dprimes(exps))
    all_dprimes[t] = result
  return all_dprimes


def calculate_dprimes(all_exps: List[MouseExp]) -> Tuple[np.ndarray,
                                                         List[float],
                                                         List[float],
                                                         List[int]]:
  """
  Calculate the d-prime for all the experiments.  Preprocess each experiment
  using the preprocess_mouse_data function.  Then calculate the d' for each
  set of experiments with the same frequency, level and channel.  Returns a
  3d array index by frequency, level, and channel.

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

  plot_num = 1
  dprimes = np.nan*np.zeros((len(all_exp_freqs), len(all_exp_levels),
                             len(all_exp_channels)))
  # Now loop through all the frequencies, channels, and levels.
  for i, freq in enumerate(all_exp_freqs):
    for k, channel in enumerate([1, 2]):
      # Find the noisy data for this combination of frequency and channel
      noise_exp = find_noise_exp(all_exps, freq=freq, channel=channel)
      if noise_exp is None:
        print(f'Found no noise data for freq={freq}, channel={channel}')
        continue
      
      noise_data = preprocess_mouse_data(noise_exp.single_trials)

      for j, level in enumerate(all_exp_levels):
        exps = find_exp(all_exps, freq=freq, level=level, channel=channel)
        if len(exps) == 0:
          print(f' Found ZERO examples for freq={freq}, level={level}, '
                f'channel={channel}: {len(exps)}')
          continue
        elif len(exps) > 1:
          print(f'  Processing {len(exps)} segments for the same preparation.')
        all_data = []
        for exp in exps:
          all_data.append(preprocess_mouse_data(exp.single_trials))
        signal_data = np.concatenate(all_data, axis=1)

        debug = channel==2 and freq==16000 and level in [0.0, 30.0, 60.0, 90.0]
        if debug:
          plt.subplot(2, 2, plot_num)
          plot_num += 1
        dprimes[i, j, k] = calculate_dprime(signal_data, noise_data, debug)
        if debug:
          plt.title(f'freq={int(freq)}, level={int(level)}, channel={int(channel)}')
  return dprimes, all_exp_freqs, all_exp_levels, all_exp_channels


def plot_dprimes(dp: DPrimeResult):
  """Create a plot summarizing the d' collected by the calculate_all_dprimes
  routine above.  Show d' versus level, for each frequency and channel pair.

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
      plt.plot(dp.levels, dp.dprimes[i, :, k],
              label=f'freq={freqs}, channel={channel}',
              linestyle=linestyle,
              color=colors[i])
  plt.title('D-Prime versus Presentation Level')
  plt.legend()
  plt.xlabel('Level (dB)');


def cache_dprime_data(d: str, 
                      dprimes: DPrimeResult,
                      dprime_pickle_name: str, 
                      load_data: bool = False):
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


###############  Cache all the Mouse CSV files ###########################

mouse_data_pickle_name = 'mouse_exp.pkl'
mouse_summary_pickle_name = 'mouse_summary.pkl'
mouse_dprime_pickle_name = 'mouse_dprime.pkl'


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


def cache_waveform_data(d: str, 
                        waveform_pickle_name: str, 
                        load_data: bool = False) -> Optional[Dict[str, 
                                                                  MouseExp]]:
  """
  Cache all the CSV files in one of George's mouse recording folders.
  If we don't have the cache file, parse all the CSV files and create
  a new cache file.  If return_data is true, return the dictionary of waveform
  data, reading it back in if we didn't compute it here.

  Args:
    exp_dir: Which data directory to read and cache.
    waveform_pickle_name: The name of the waveform cache file.
    load_data: Whether to return the cached data if it is there.

  Returns:
    A list of MouseExp structures.
  """
  pickle_file = os.path.join(d, waveform_pickle_name)
  if not os.path.exists(pickle_file):
    try:
      print(f'  Reading mouse waveforms from {d}')
      all_trials = read_all_mouse_dir(d, debug=True)
      with open(pickle_file, 'w') as f:
        f.write(jsonpickle.encode(all_trials))
        print(f'  Cached {len(all_trials)} experiments')
    except Exception as e:
      print(f'  **** Could not read {d} because of {repr(e)}. Skipping')
      return None
  if load_data:
    with open(pickle_file, 'r') as f:
      all_trials = jsonpickle.decode(f.read())
      if all_trials:
        print(f'  Loaded {len(all_trials)} waveforms from {pickle_file}.')
      else:
        print(f'  Found empty pickle file in {pickle_file}')
  return all_trials


def summarize_all_data(all_exp_dirs: List[str], pickle_name):
  for d in all_exp_dirs:
    try:
      print(f'Summarizing data in {d}')
      with open(os.path.join(d, pickle_name), 'r') as f:
        all_exps = jsonpickle.decode(f.read())
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


###############  Analyze all the d' data ##############################


def get_all_dprime_data(dirs):
  all_dprimes = {}
  for d in dirs:
    animal_date = os.path.basename(d)
    pickle_file = os.path.join(d, 'mouse_dprimes.pkl')
    if os.path.exists(pickle_file):
      with open(pickle_file, 'rb') as fp:
        mouse_dprime = jsonpickle.decode(fp.read())
      mouse_dprime2 = {}
      for k in mouse_dprime.keys():
        mouse_dprime2[f'{animal_date}_{k}'] = mouse_dprime[k]
      all_dprimes.update(mouse_dprime2)
  return all_dprimes


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
                  plot=False) -> None:
  """Add the SPL threshold to a DPrimeResult.  This is done by fitting a
  positive-coeffifient polynomial to the d' data at each frequency and channel,
  and then using this smooth model to predict the level where d' exceeds the
  desired level.
  Args:
    dprimes_result: Consolidated estimate of the d' for each frequency, level
      and channel
    dp_criteria: Desired d' threshold
    plot: Generate plot showing curve fits
  Returns:
    Nothing.  dprimes_result is modified in place.
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
  dprimes = dprimes_result.dprimes
  for i, freq in enumerate(dprimes_result.freqs):
    for j, channel in enumerate(dprimes_result.channels):
      levels = dprimes_result.levels
      dprimes = dprimes_result.dprimes[i, :, j]
      cp = None
      try:
        cp = PositivePolynomial()
        cp.fit(levels, dprimes)
        r = cp.threshold(dp_criteria)
        # Check if r is a list and take the first element if it is
        # This ensures we store a single numeric value in db_at_threshold
        if isinstance(r, list):
            r = r[0] if r else np.nan
      except Exception:
        print('Could not fit polynomial')
        r = np.nan
      spl_threshold[i, j] = r
      smoothed[i, :, j] = cp.eval(np.asarray(dprimes_result.levels))
      if cp and plot:
        if channel == 1:
          ls = '--'
        else:
          ls = '-'
        plt.plot(plot_levels, [cp.eval(l) for l in plot_levels], 
                label=f'{channel_names[channel]} at {freq}Hz',
                color=color_list[i], ls=ls)
        plt.plot(levels, dprimes, 'x', 
                color=color_list[i])
  if plot:
    plt.legend()
    plt.xlabel('Sound Level (dB)')
    plt.ylabel('d\'')
  dprimes_result.spl_threshold = spl_threshold
  dprimes_result.smooth_dprimes = smoothed


def add_all_thresholds(all_dprimes: Dict[str, DPrimeResult], dp_criteria=2):
  for k in all_dprimes.keys():
    add_threshold(all_dprimes[k], dp_criteria=dp_criteria)


def accumulate_thresholds(all_dprimes: Dict[str, DPrimeResult],
                          freqs: Optional[List[float]] = None,
                          max_spl=120) -> Tuple[List[float], 
                                                                 List[float]]:
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
  all_abr = np.concatenate(all_abr)
  all_ecog = np.concatenate(all_ecog)
  good = np.logical_and(np.logical_and(np.isfinite(all_abr), all_abr < max_spl),
                        np.logical_and(np.isfinite(all_ecog), all_ecog < max_spl))
  abr_thresh = all_abr[good]
  ecog_thresh = all_ecog[good]
  pearson_r = spstats.pearsonr(abr_thresh, ecog_thresh).statistic
  return all_abr, all_ecog, pearson_r


def plot_threshold_scatter(abr_thresh, ecog_thresh, min_abr_threshold=np.nan,
                           color='b', draw_unit=True):
  plt.plot(abr_thresh, ecog_thresh, 'x', color=color)
  plt.xlabel('ABR (Channel 1) Threshold (dB)')
  plt.ylabel('ECoG (Channel 2) Threshold (dB)')
  plt.title(f'Comparison of Threshold at d\'={min_abr_threshold}')
  plt.xlim(0, 120)
  plt.ylim(0, 120)
  if draw_unit:
    a=max(plt.axis())
    plt.plot([0, a], [0, a], '--');


###############  Main program, so we can run this offline ######################

FLAGS = flags.FLAGS
flags.DEFINE_string('basedir',
                    'drive/Shareddrives/StanfordAudiology/GeorgeMouseABR/CAP_ABR',
                    'Base directory to find the ABRPresto mouse data')
flags.DEFINE_string('waveforms_cache', 'mouse_exp.pkl',
                    'Where to cache all the waveforms in this directory')
flags.DEFINE_string('dprimes_cache', 'mouse_dprimes.pkl',
                    'Where to cache the dprimes in this directory')
flags.DEFINE_string('filter', '', 'Which directories to process, ignore rest.')

def process_one_dir(dir, waveform_cache, dprime_cache):
  if (os.path.exists(waveform_cache) and os.path.getsize(waveform_cache) and
    os.path.exists(dprime_cache) and os.path.getsize(dprime_cache)):
    print(f'Skipping waveforms and dprimes in {dir} because they are '
          'already cached.')
    return
  print(f'Processing waveforms in {dir}')
  all_exps = cache_waveform_data(dir, waveform_cache, True)
  if all_exps:
    dprimes = calculate_all_dprimes(all_exps)
    cache_dprime_data(dir, dprimes, dprime_cache)
  else:
    print(f'  No waveform data to process for dprimes.')

def main(_):
  all_mouse_dirs = find_all_mouse_directories(FLAGS.basedir)
  all_dprimes = {}
  for dir in all_mouse_dirs:
    if FLAGS.filter in dir:
      waveform_cache = os.path.join(dir, FLAGS.waveforms_cache)
      dprime_cache = os.path.join(dir, FLAGS.dprimes_cache)
      process_one_dir(dir, waveform_cache, dprime_cache)

if __name__ == '__main__':
  app.run(main)

