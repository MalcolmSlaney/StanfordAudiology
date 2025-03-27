# Code to support analysis of George's mouse ABR/ECoG recordings.  This colab
# shows how to use this code:
# https://colab.research.google.com/drive/1wtTeslQa8BQIk9QxUfOJawU6AuhvmaDf
import csv
import dataclasses
import glob
import math
import os
import subprocess
import traceback

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spstats
from scipy.optimize import curve_fit

from absl import app
from absl import flags
from abr import *  # Import this first by hand
from abr import remove_offset
from typing import Dict, List, Optional, Union, Tuple

# We save the raw data with Pickle because raw JSON doesn't support Numpy
# https://jsonpickle.github.io/
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

"""
File progression:

George's data is stored in a Google drive folder, one subfolder per recording
date, such as 20230720.

For each date, there are a number of .csv files, one per experimental
condition.  A condition is the recording circumstance (pre/post), as well as
the frequency, level and channel number.  The CSV file contains (after the two
rows of header) a single row per recording at 2414Hz.  There are multiple rows,
one per stimulus.

All these trials are consolidated into one or more
  mouse_waveforms01.pkl
files (since the CSV files are expensive to read).  The contents of these
pickle files are MouseExp's.  Each of these files are stored in the
corresponding date directory.

We read all the waveforms and calculate the covariance d' and store the
results, one per day in a
  mouse_dprime.pkl
The data here is stored as a dictionary pointing to DPrimeResult classes.  One
file per date directory.

There are two types of summary files, across all dates.

There are a number of files to consolidate the waveforms from "good" trials
together.  They are in the main directory, and have names of the form
  good_waveform_cache00.pkl.
These files consist of lists of MouseExp classes.

Finally, all the RMS estimates from "good" experiments are stored in a file
called
  rms_waveform_cache.pkl
This data is a List (by frequency) of List (by level) of List (by channel) of
list of RMS values (one per animal)

"""
################### Waveform Level Reading/Caching/Loading ##################


@dataclasses.dataclass
class MouseExp:
  """
  A data structure that describes one set of ABR experimental data, indexed by
    frequency, level, and channel.  The basename field describes the type of
    experiment (pre, post, etc.)

  Attributes:
    filename: Where the data came from, full path
    basename: The date and experiment type.  For example
      20230706_control1_ABRCAP_post10min-0-24-1-1.csv
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
  freq: float    # Hz
  level: float   # dB
  channel: int   # which electrode, probably 1 or 2
  sgi: int       # TDT Stimulus generation index (i.e. freq and level)
  description: str = ''
  single_trials: np.ndarray = None  # num_waveform samples x num_trials
  paired_trials: np.ndarray = None


mouse_sample_rate = 24414*8  # From George's Exp Notes, 8x oversampling
# Existing hardware filtering from 2.2-7500Hz.


def read_mouse_exp(filename: str) -> MouseExp:
  """
  Read a CSV file containing data about one ABR experiment.  Parse the header
  lines, and then read in the table of data.

  Args:
    filename: The full file-system path to the CSV file we want to read.

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
      if len(row) > 9:  # Arbitrary
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

mouse_waveforms_pickle_name = 'mouse_waveforms_v2.pkl'
mouse_dprimes_pickle_name = 'mouse_dprime_v2.pkl'


def find_all_mouse_directories(mouse_data_dir: str) -> List[str]:
  """Look for all the directories that seem to contain George's mouse data.
  Walk the directory tree starting at the given directory, looking for all
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


def waveform_cache_present(
    dir: str, waveform_pickle_name: str = mouse_waveforms_pickle_name):
  if os.path.exists(os.path.join(dir, waveform_pickle_name)):
    return True
  new_filename = waveform_pickle_name.replace('.pkl', '00.pkl')
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
  """The CSV files are converted into a small number of PKL files for easier
  loading. (CSV is slow.) This routine reads in all the cached data (from
  multiple files) in one directory.)
  """
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
        print('  No experiments.')
      else:
        all_sizes = [str(e.single_trials.shape) for e in all_exps]
        all_sizes = ', '.join(all_sizes)
        print(f'  Sizes: {all_sizes}')
        print('  Channels: '
              f'{sorted(list(set([e.channel for e in all_exps])))}')
        print('  Frequencies: '
              f'{sorted(list(set([e.freq for e in all_exps])))}')
        # break
    except Exception as e:
      print(f'  Could not load mouse data for {d} because of {e}')


def cache_all_mouse_dir(
    expdir: str,
    waveform_pickle_name: str = mouse_waveforms_pickle_name,
    max_files: int = 0,
    max_bytes: float = 10e9,
    debug: bool = False) -> None:
  """
  Read and cache the CSV mouse experiments in the given directory. Each
  trial experiment is stored in a single CSV file.  This routine reads all the
  csv files and converts them into numpy arrays, stored as pickle files.

  Args:
    expdir:  Where to find the experimental for this animal
    waveform_pickle_name: The canonical name for the resulting pickle files
      (They will be numbered later.)
    max_files: How many files to put into one pickle file, to limit for
      debugging.
    max_bytes: How many bytes, about, to put in each file.  Will be one
      experiment bigger than this.
    debug: Extra summary print statements.

  Returns:
    List of MouseExp structures summarizing the data in this directory.
  """
  def cache_size(all_trials: List[MouseExp]):
    return sum([exp.single_trials.nbytes for exp in all_exps])

  print('Cache_all_mouse_dir:', expdir, max_files, max_bytes)
  all_exp_files = [f for f in os.listdir(expdir)
                   if os.path.isfile(os.path.join(expdir, f)) and
                   f.endswith('.csv')]

  all_exps = []
  cache_file_count = 0
  total_file_count = len(all_exp_files)
  for f in all_exp_files:
    if 'saline' in f or 'test' in f:
      print(f'Skipping unneeded condition: {f}')
      continue
    if debug:
      print(f'    Reading {f} ({len(all_exps)}/{total_file_count} '
            f'totaling {cache_size(all_exps)} bytes)')
    exp = read_mouse_exp(os.path.join(expdir, f))
    all_exps.append(exp)
    if max_files and len(all_exps) >= max_files:
      print(f'  Reached maximum limit of {max_files} files to process.')
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
             freq: Optional[float] = None,
             level: Optional[float] = None,
             channel: Optional[int] = None) -> List[MouseExp]:
  """
  Find particular experiments in the list of all experiments.

  Args:
    all_exps: a list containing experiments in MouseExp format
    freq: desired frequency (None means any freq)
    level: desired level (None means any level)
    channel: Recording channel (normal ABR, ECochG)

  Returns:
    A list of MouseExp's with the desired frequency and level.
  """
  good_ones = []
  for exp in all_exps:
    if freq is not None and freq != exp.freq:
        continue
    if level is not None and level != exp.level:
        continue
    if channel is not None and channel != exp.channel:
        continue
    good_ones.append(exp)
  return good_ones


def find_noise_exp(all_exps: List[MouseExp],
                   freq: Optional[float] = None,
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
                          mouse_sample_rate: float = 24414*8,
                          first_sample: int = 0,
                          last_sample: int = -1) -> np.ndarray:
  """
  Preprocess the mouse data, removing the DC offset, rejecting artifacts, and
  applying a bandpass filter.

  Args:
    data: A matrix of shape num_samples x num_trials, opposite of what the rest
      of the routines that follow need.
    remove_dc: the average value of each trial, an easy low-pass filter
    remove_artifacts: ???
    bandpass_filter: Do we bandpass filter the data?
    low_filter: If bandpass filtering the data, the low-frequency cutoff (Hz)
    high_filter: If bandpass filtering the data, the high-frequency cutoff (Hz)
    mouse_sample_rate: The sample rate for the data
    first_sample: Extract part of the waveform data, starting with this sample
    last_sample: Extract part of the waveform data, ending with this sample (-1
      means all the data)

  Returns:
    A matrix of shape num_samples x num_trials, transposed from the original.
  """
  if remove_dc:
    data = remove_offset(data)  # Now data is time x num_trials
  if remove_artifacts:
    data = reject_artifacts(data)
  if bandpass_filter:
    # Bidelman used 90-2000?
    data = butterworth_filter(data, lowcut=low_filter, highcut=high_filter,
                              fs=mouse_sample_rate, order=6, axis=0)
  if last_sample == -1:
    last_sample = data.shape[0]
  # print(f'Returning {first_sample} to {last_sample} of {data.shape} giving '
  #       f'{data[first_sample:last_sample, :].shape}')
  return data[first_sample:last_sample, :]


def shuffle_data(data: np.ndarray, axis=0) -> np.ndarray:
  """
  Shuffle the data in time.

  Args:
    data: A matrix of shape num_samples x num_trials

  Returns:
    A shuffled copy of the data.
  """
  rng = np.random.default_rng()  # Create a random number generator instance.
  # Make sure to copy the input data because shuffle rearranges its argument.
  data = data.copy()
  rng.shuffle(data, axis=axis)  # Shuffle in time
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
    A dictionary of groups of MouseExps.  The key is the experiment type and
    the value is a list of MouseExps.
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
  cov_dprimes: np.ndarray     # 3d array by frequency, level and channel
  rms_of_total: np.ndarray    # RMS values for total signals
  rms_of_average: np.ndarray  # RMS values for the average of each group
  rms_dprimes: np.ndarray     # Corresponding d' for the RMS calculations.
  freqs: List[float]
  levels: List[float]
  channels: List[int]
  # The smooth arrays are derived by smoothing the raw data above and have the
  # same dimensions as the primary arrays above.
  # The threshold arrays are derived from the primary arrays above, and our
  # 2D arrays (removed the level dimension).
  dp_criteria: float = -1  # The d' criteria used for thresholds below
  cov_spl_threshold: Optional[np.ndarray] = None
  cov_smooth_dprimes: Optional[np.ndarray] = None
  rms_spl_threshold: Optional[np.ndarray] = None
  rms_smooth_dprimes: Optional[np.ndarray] = None

  def check(self) -> None:
    """Check the data arrays in a DPrimeExp object to make sure that their
    sizes are all consistent.  Throws an error if not.
    """
    num_freqs = len(self.freqs)
    num_levels = len(self.levels)
    num_channels = len(self.channels)

    assert self.cov_dprimes.shape == (num_freqs, num_levels, num_channels)
    assert self.rms_of_total.shape == (num_freqs, num_levels, num_channels)
    assert self.rms_of_average.shape == (num_freqs, num_levels, num_channels)
    assert self.rms_dprimes.shape == (num_freqs, num_levels, num_channels)

    if self.cov_spl_threshold:
      assert self.cov_spl_threshold.shape == (num_freqs, num_channels)
    if self.cov_smooth_dprimes:
      assert self.cov_smooth_dprimes.shape == (num_freqs, num_channels)
    if self.rms_spl_threshold:
      assert self.rms_spl_threshold.shape == (num_freqs, num_channels)
    if self.rms_smooth_dprimes:
      assert self.rms_smooth_dprimes.shape == (num_freqs, num_channels)

  def add_threshold(self, dp_criteria=2,
                    fit_method: str = 'bilinear',
                    plot=False) -> None:
    """Add the SPL threshold to a DPrimeResult.  This is done by fitting a
    either a polynomial or bilinear model to the d' data at each frequency and
    channel, and then using this smooth model to predict the level where d'
    exceeds the desired level.
    Args:
      self: Consolidated estimate of the d' for each frequency, level
        and channel
      dp_criteria: Desired d' threshold
      fit_method: Either 'bilinear' or 'polynomial'
      plot: Generate plot showing curve fits
    Returns:
      Nothing. This object is modified in place, adding:
        1) the spl_threshold where d' gets above the dp_criteria, and
        2) smoothed_dprimes, which gives the d' estimate, after polynomial
          smoothing at the same frequencies, levels, and channels as the
          incoming dprime array.
    """
    if len(self.channels) != 2:
      return np.zeros((0, 2))
    spl_threshold = np.zeros((len(self.freqs),
                              len(self.channels)))
    min_level = min(self.levels)
    max_level = max(self.levels)
    plot_levels = np.linspace(min_level, max_level, 100)
    channel_names = ['', 'ABR', 'ECochG']

    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    smoothed = np.zeros((len(self.freqs),
                        len(self.levels),
                        len(self.channels)))
    dprimes = self.cov_dprimes
    for i, freq in enumerate(self.freqs):
      for j, channel in enumerate(self.channels):
        levels = self.levels
        dprimes = self.cov_dprimes[i, :, j]
        try:
          if fit_method == 'bilinear':
            interp = BilinearInterpolation()
          elif fit_method == 'polynomial':
            interp = PositivePolynomial()
          else:
            assert ValueError(f'Unknown fit method: {fit_method}')
          interp.fit(levels, dprimes)
          thresh_db = interp.threshold(dp_criteria)
          # Check if thresh_db is a list and take the first element if it is
          # This ensures we store a single numeric value in db_at_threshold
          if isinstance(thresh_db, list):
              thresh_db = thresh_db[0] if thresh_db else np.nan
        except Exception as error:
          print(f'Could not fit levels: {error}')
          print(traceback.format_exc())
          thresh_db = np.nan
        spl_threshold[i, j] = thresh_db
        smoothed[i, :, j] = interp.eval(np.asarray(self.levels))
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
          plt.axvline(thresh_db, color='r', ls=':')  # one line per freq
    if plot:
      plt.legend()
      plt.xlabel('Sound Level (dB)')
      plt.ylabel('d\'')
      plt.xlim(0, 100)
    self.cov_spl_threshold = spl_threshold
    self.cov_smooth_dprimes = smoothed
    self.dp_criteria = dp_criteria


###############  Summarize and smooth the d' data ############################


def get_all_dprime_data(
    dirs: List[str],
    pickle_name: str = mouse_dprimes_pickle_name) -> Dict[str, DPrimeResult]:
  """Get all the d' data from the list of directories of mouse ABR data.

  Args:
    dirs: All the directories from which to read the cached d' data.
    pickle_name: What is the name of the pickle file in each directory.

  Returns:
    A dictionary mapping experiment name to d' data.
  """
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
    # Make sure ydata is increasing, not perfect but better than noise
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
    ydata = np.maximum.accumulate(self._ydata)
    if y <= ydata[0]:
      i = 0
    elif y >= ydata[-2]:
      i = len(ydata)-2
    else:
      i = np.nonzero(y > ydata)[0][-1]
    assert i >= 0
    assert i <= len(ydata)-2, f'i too big: y={y}, ydata={ydata}, i={i}'
    delta = (y - ydata[i])/(ydata[i+1]-ydata[i])
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
    a = self._a - threshold  # Subtract threshold so we can find zero crossings
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


def add_all_thresholds(all_dprimes: Dict[str, DPrimeResult], dp_criteria=2,
                       fit_method='bilinear') -> None:
  """Process all the dprime structures we have, using the add_threshold
    function.

  Args:
    all_dprimes: Dictionary pointing to all the d' data that we have.
    dp_criteria: How high does the d' have to be to pass this arbitrary
      threshold level (defaults to 2)
    fit_method: How do we interpolate the d' versus level to find the point
      when the d' data crosses the threshold above.

  Return:
    Nothing returned, all d' objects modified in place.
  """
  for k in all_dprimes.keys():
    all_dprimes[k].add_threshold(dp_criteria=dp_criteria,
                                 fit_method=fit_method)


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


def calculate_dprime(h1: Union[list, np.ndarray],
                     h2: Union[list, np.ndarray],
                     geometric_mean: bool = False) -> float:
  """Calculate the d' given two sets of (one-dimensiona) data.  The h1
  data should be the bigger of the two data. The normalization factor either
  the arithmetic mean (default) of the two standard deviations, if the data is
  additive, or a geometric mean if the data is based on a multiplicative ratio.
  """
  if geometric_mean:
    return (np.mean(h1) - np.mean(h2)) / np.sqrt(np.std(h1)*np.std(h2))
  else:
    # Normalize by arithmetic mean of variances (not std)
    norm = np.sqrt((np.std(h1)**2 + np.std(h2)**2)/2.0)
    return (np.mean(h1) - np.mean(h2)) / norm


def calculate_cov_dprime(data: np.ndarray,
                         noise_data: Optional[np.ndarray] = None,
                         with_self_similar: bool = False,
                         debug: bool = False,
                         theoretical_model: Optional[np.ndarray] = None,
                         score_loc: Union[bool, Tuple] = True) -> float:
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
      If not specified shuffle the data matrix.
    with_self_similar: Whether to include the present trial in the model.
      Originally we did this, but this inflates the d' calculation for small
      number of trials since the self-similar component scores very well.
    debug: Whether to show a plot of the histogram
    theoretical_model: Use this model instead of computing it from the data.
      Only use when with_self_similar is True.
    score_loc: Where to put a legend above the scores
      True: automatic on the left
      False: No score legend
      Two-ple: x and y locations in plot coordinates
  Returns:
    A scalar representing the d-prime.
  """
  if noise_data is None:
    noise_data = data
  shuffled_data = shuffle_data(noise_data)

  if theoretical_model is None:
    model = np.mean(data, axis=1) #, keepdims=True)
  else:
    assert len(theoretical_model) == data.shape[0]
    model = theoretical_model

  if with_self_similar:
    h1 = np.reshape(model, (-1, 1)) * data
    h1_response = np.sum(h1, axis=0)  # Sum response over time
  else:
    num_trials = data.shape[1]
    h1_response = np.zeros(num_trials)
    for i in range(num_trials):
      model_without = (model*num_trials - data[:, i])/(num_trials-1)
      h1_response[i] = np.sum(model_without * data[:, i], axis=0)
  h2 = np.reshape(model, (-1, 1)) * shuffled_data
  h2_response = np.sum(h2, axis=0)  # Sum response over time
  dprime = calculate_dprime(h1_response, h2_response)
  if debug:
    data_range = (min(np.min(h1_response), np.min(h2_response)),
             max(np.max(h1_response), np.max(h2_response)))
    counts, bins = np.histogram(h1_response, bins=40, range=data_range)
    plt.plot((bins[:-1]+bins[1:])/2.0, counts, label='signal_trial')
    counts, bins = np.histogram(h2_response, bins=40, range=data_range)
    plt.plot((bins[:-1]+bins[1:])/2.0, counts, label='noise trial')
    plt.legend()
    plt.title('Histogram of covariance')
    if score_loc:
      if score_loc is True:
        xloc, _, yloc, _ = plt.axis()
      elif isinstance(score_loc, (tuple, list)) and len(score_loc) == 2:
        xloc, yloc = score_loc
      plt.text(xloc, yloc,
               f' H1: {np.mean(h1_response):4.3G} +/- '
               f'{np.std(h1_response):4.3G}\n'
               f' H2: {np.mean(h2_response):4.3G} +/-'
               f'{np.std(h2_response):4.3G}\n'
               f' d\'={dprime:4.3G}\n\n\n')
  return dprime


def calculate_dprime_by_trial_count_bs(filtered_abr_stack: np.ndarray,
                                       level_index = 9,
                                       noise_index = 0,
                                       freq_index = 1,
                                       channel_index = 1,
                                       min_count = 20,
                                       max_count = 20000,
                                       repetition_count: int = 20,
                                       with_self_similar: bool = True,
                                       num_divisions: int = 10
                                       ) -> Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray]:
  # The shape of the stacks array is Freqs x levels x channels x time x trials
  # Use bootstrapping this time
  assert filtered_abr_stack.ndim == 5
  assert level_index < filtered_abr_stack.shape[1]
  assert noise_index < filtered_abr_stack.shape[1]
  assert freq_index < filtered_abr_stack.shape[0]
  assert channel_index < filtered_abr_stack.shape[2]

  time_sample_count = filtered_abr_stack.shape[3]
  trial_count = filtered_abr_stack.shape[4]

  block_sizes = (trial_count / (2**np.arange(0,
                                             num_divisions, 1.0))).astype(int)
  block_sizes = block_sizes[(block_sizes >= min_count) &
                            (block_sizes <= max_count)]
  dprime_mean_by_size = np.zeros(len(block_sizes))
  dprime_std_by_size = np.zeros(len(block_sizes))

  for i, block_size in enumerate(block_sizes):
    dps = []
    for j in range(repetition_count):
      # Note: transpose the resulting array slices because of this answer:
      #  https://stackoverflow.com/a/71489304
      signal_data = filtered_abr_stack[freq_index, level_index,
                                       channel_index, :,
                                       np.random.choice(trial_count,
                                                        block_size)].T
      noise_data = filtered_abr_stack[freq_index, noise_index,
                                      channel_index, :,
                                      np.random.choice(trial_count,
                                                       block_size)].T
      dps.append(calculate_cov_dprime(signal_data, noise_data,
                                      with_self_similar=with_self_similar,
                                      debug=False))
    dprime_mean_by_size[i] = np.mean(dps)
    dprime_std_by_size[i] = np.std(dps)
  return block_sizes, dprime_mean_by_size, dprime_std_by_size


def calculate_rmses(signal_data, noise_data, debug):
  noise_rms = calculate_rms(noise_data)
  signal_rms = calculate_rms(signal_data)
  dprime = calculate_dprime(signal_rms, noise_rms)

  rms_of_signal = np.sqrt(np.mean(signal_data**2))
  rms_of_average = np.sqrt(np.mean(np.mean(signal_data, axis=-1)**2))

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

  return rms_of_signal, rms_of_average, dprime


def calculate_all_summaries(all_exps: List[MouseExp],
                            first_sample: int = 0,
                            last_sample: int = -1,
                            ) -> Dict[str, DPrimeResult]:
  """Calculate the waveform summaries for each type of experiment within this
  list of results.  Each result is for one experiment, at one frequency, level
  and channel. This code groups the experiments together that share the same
  type, based on the second component of the file name, and then computes the
  d' as a function of frequency, level and channel.

  Arg:
    all_exps: A list of MouseExps, containing the raw waveform data for each
      condition
    first_sample: Extract part of the waveform data, starting with this sample
    last_sample: Extract part of the waveform data, ending with this sample (-1
      means all the data)

  Returns:
    A dictionary, keyed by experiment type, containing the dprime result.
  """
  all_groups = group_experiments(all_exps)
  all_dprimes = {}
  for t, exps in all_groups.items():
    result = DPrimeResult(*calculate_waveform_summaries(exps,
                                                        first_sample,
                                                        last_sample))
    all_dprimes[t] = result
  return all_dprimes


def calculate_waveform_summaries(all_exps: List[MouseExp],
                                 debug_cov_not_rms: bool = True,
                                 debug_freq: Optional[float] = None,
                                 debug_levels: List[float] = [],
                                 debug_channel: Optional[int] = None,
                                 first_sample: int = 0,
                                 last_sample: int = -1,
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
    first_sample: Extract part of the waveform data, starting with this sample
    last_sample: Extract part of the waveform data, ending with this sample (-1
      means all the data)

  Returns:
    A tuple consisting of the following items.  They must be in the *same*
    order as the fields in the DPrimeResult class object.  All 3d arrays have
    shape (freq x levels x channels):
      a 3d array of d' for the covariance measure, for each experiment,
      a 3d array of RMS values for the total signal (rms_of_total),
      a 3d array of RMS values for the average of trial type (rms_of_average),
      a 3d array of d' for the RMS measure
      Then lists of the found the corresponding frequences, levels, and
      channels
    The order of these results is be the same as the fields in DPrimeExp class.
  """
  all_exp_levels = sorted(list(set([exp.level for exp in all_exps])))
  all_exp_freqs = sorted(list(set([exp.freq for exp in all_exps])))
  all_exp_channels = sorted(list(set([exp.channel for exp in all_exps])))

  plot_num = 1
  cov_dprimes = np.nan*np.zeros((len(all_exp_freqs), len(all_exp_levels),
                                len(all_exp_channels)))
  rms_of_signals = cov_dprimes.copy()
  rms_of_averages = cov_dprimes.copy()
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

      noise_data = preprocess_mouse_data(noise_exp.single_trials,
                                         first_sample, last_sample)
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
          all_data.append(preprocess_mouse_data(exp.single_trials,
                                                first_sample, last_sample))
        signal_data = np.concatenate(all_data, axis=1)

        debug = (channel == debug_channel and freq == debug_freq and
                 level in debug_levels)
        if debug:
          plt.subplot(2, 2, plot_num)
          plot_num += 1
        cov_dprimes[i, j, k] = calculate_cov_dprime(signal_data, noise_data,
                                                    debug = (debug and 
                                                             debug_cov_not_rms))
        (rms_of_signal, rms_of_average,
         dprime) = calculate_rmses(signal_data, noise_data,
                                   debug and not debug_cov_not_rms)

        signal_rms = calculate_rms(signal_data)
        rms_of_signals[i, j, k] = rms_of_signal
        rms_of_averages[i, j, k] = rms_of_average
        rms_dprimes[i, j, k] = dprime
        if debug:
          plt.title(f'freq={int(freq)}, level={int(level)}, '
                    f'channel={int(channel)}')
  print(f'  Processed {all_processed} CSV files, {all_multiprocessed} '
        'part of a group.')
  return (cov_dprimes, rms_of_signals, rms_of_averages, rms_dprimes,
          all_exp_freqs, all_exp_levels, all_exp_channels)


def filter_dprime_results(all_dprimes: Dict[str, DPrimeResult],
                          keep_list: List[str] = [],
                          drop_list: List[str] = [],
                          abr_thresh_greater_than: float = 0.0,   # Min d' threshold
                          abr_thresh_less_than: float = 1e9,   # Max d' threshold
                          min_ecog_thresh: float = 0.0,  # Min d' threshold
                          ecog_thresh_less_than: float = 1e9,  # Max d' threshold
                          abr_resp_greater_than: float = 0.0,
                          abr_resp_less_than: float = 1e9,
                          ecog_resp_greater_than: float = 0.0,
                          ecog_resp_less_than: float = 1e9,
                          ) -> Dict[str, DPrimeResult]:
  """Filter a DPrime dictionary, looking for good preparations and dropping the
  bad ones.  And setting limits on the calculate thresholds, looking for good
  and bad data.  Be sure to run add_all_thresholds() before running this filter
  command.

  Args:
    all_dprimes: A dictionary pointing to dprime results
    keep_list: A list of strings with words from the date_preparation_name keys
      that we want to keep
    drop_list: Like keep list, but overrules with keys to drop
    abr_thresh_greater_than: Keep preps where all thresholds are *above* this limit
    abr_thresh_less_than: Keep preps where all thresholds are *below* this limit
    min_ecog_thresh: Like abr_thresh_greater_than, but using ECoG data
    ecog_thresh_less_than: Like abr_thresh_less_than, but using ECoG data

    abr_resp_greater_than: Test whether highest level is above this d'
    abr_resp_less_than: Test whether all levels are below this d'
    ecoh_resp_greater_than: Test whether highest level is above this d'
    ecoh_resp_less_than: Test whether all levels are below this d'
  Returns:
    A new dictionary containing just the selected dprime results.
  """
  filtered_dprimes = {}
  for k in all_dprimes.keys():
    if any([l in k for l in drop_list]):
      continue
    if len(keep_list) and not any([l in k for l in keep_list]):
      continue

    dp = all_dprimes[k]
    if dp.cov_spl_threshold is None:
      continue
    if (not isinstance(dp.cov_spl_threshold, np.ndarray) or
        dp.cov_spl_threshold.ndim < 2):
      continue

    if np.all(np.isnan(dp.cov_spl_threshold[:, 0])):  # No ABR Data
      continue
    if np.all(np.isnan(dp.cov_spl_threshold[:, 1])):  # No ECochG Data
      continue    
    if np.nanmin(dp.cov_spl_threshold[:, 0]) < abr_thresh_greater_than:
      continue
    if np.nanmax(dp.cov_spl_threshold[:, 0]) > abr_thresh_less_than:
      continue
    if np.nanmin(dp.cov_spl_threshold[:, 1]) < min_ecog_thresh:
      continue
    if np.nanmax(dp.cov_spl_threshold[:, 1]) > ecog_thresh_less_than:
      continue
    if np.min(dp.cov_dprimes[:, -1, 0]) < abr_resp_greater_than:
      continue
    if np.max(dp.cov_dprimes[:, :, 0]) > abr_resp_less_than:
      continue
    if np.min(dp.cov_dprimes[:, -1, 1]) < ecog_resp_greater_than:
      continue
    if np.max(dp.cov_dprimes[:, :, 1]) > ecog_resp_less_than:
      continue

    filtered_dprimes[k] = dp
  return filtered_dprimes


def plot_dprimes(dp: DPrimeResult, plot_cov_dp: bool = True, 
                 show_threshold: bool = False,
                 title: str = ''):
  """Create a plot summarizing the d' of the covariance data collected by the
  calculate_all_summaries routine above.  Show d' versus level, for each
  frequency and channel pair.

  Args:
    dprimes: a 3d array of dprimes, for all experiments, as a function of
      frequences, levels, and channels.
    plot_cov_dp: If true, plot covariance d', otherwise RMS d'
  """
  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color']

  names = ['', 'ABR', 'ECochG']
  if plot_cov_dp:
    data = dp.cov_dprimes
    thresh = dp.cov_spl_threshold
    title = title or 'Covariance D-Prime versus Presentation Level'
  else:
    data = dp.rms_dprimes
    thresh = dp.rms_spl_threshold
    title = title or 'RMS D-Prime versus Presentation Level'
  for i, freq in enumerate(dp.freqs):
    for k, channel in enumerate(dp.channels):
      if channel == 2:
        linestyle = '--'
      else:
        linestyle = '-'
      if show_threshold and isinstance(thresh, np.ndarray):
        thresh_label = f' Threshold={thresh[i, k]:6.0f}dB'
      else:
        thresh_label = ''
      plt.plot(dp.levels, data[i, :, k],
               label=f'{names[channel]} {freq}Hz{thresh_label}',
               linestyle=linestyle,
               color=colors[i])
  plt.title(title)
  plt.legend()
  plt.xlabel('Level (dB)')
  plt.ylabel('d\'')


def compute_and_cache_dprimes():
  cmd = 'StanfordAudiology/abr_george.py'
  run_local = True

  mouse_dirs = [os.path.basename(m) for m in
                find_all_mouse_directories(GeorgeMouseDataDir)]
  # mouse_dirs = ['20230828']
  for dir in mouse_dirs:
    base = os.path.basename(dir)
    full_dir = os.path.join(GeorgeMouseDataDir, dir)
    if not os.path.exists(full_dir):
      continue
    print(dir)
    waveform_file = os.path.join(full_dir, mouse_waveforms_pickle_name)
    dprime_file = os.path.join(full_dir, mouse_dprimes_pickle_name)
    if (os.path.exists(waveform_file) and os.path.getsize(waveform_file) and
        os.path.exists(dprime_file) and os.path.getsize(dprime_file)):
      print(f'Skipping {base}')
      continue
    if run_local:
      # cache_waveform_one_dir(full_dir, mouse_waveforms_pickle_name,
      #                        max_bytes=5*1e9)
      cache_dprime_one_dir(full_dir, mouse_waveforms_pickle_name,
                           mouse_dprimes_pickle_name)
    else:
      out = subprocess.run(['/usr/bin/python3', cmd, f'--filter={base}',
                            f'--basedir={full}'],
                           capture_output=True)
      print(out.stdout.decode('utf-8'))
      print(out.stderr.decode('utf-8'))
    print()


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


def accumulate_all_thresholds(all_dprimes: Dict[str, DPrimeResult],
                              freqs: Optional[List[float]] = None,
                              max_spl=120) -> Tuple[List[float],
                                                    List[float], float]:
  """Accumulate all the thresholds for ABR and ECoG data, across all the data
  we have.  Filter out the preparation names we do and don't want.  And remove
  remove any preparations where the computed threshold is greater than max_spl,
  indicating that we got no data for this prep.

  Args:
    all_dprimes: Dictionary keyed by the date_prep_name and pointing to the
      d' data for this preparation.  In particular we look at the
      spl_threshold, which should already be calculated by add_threshold()
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
    if dp.cov_spl_threshold is None:
      continue
    if freqs is not None:
      if not isinstance(freqs, list):
        freqs = [freqs,]
      freq_indices = [dp.freqs.index(f) for f in freqs if f in dp.freqs]
    else:
      freq_indices = range(len(dp.freqs))
    # print(freq_indices, dp.cov_spl_threshold.shape)
    all_abr.append(dp.cov_spl_threshold[freq_indices, 0].flatten())
    all_ecog.append(dp.cov_spl_threshold[freq_indices, 1].flatten())
  if len(all_abr) == 0:
    print(f'Found no data for frequency list: {",".join(freqs)}')
    return [], [], 0.0
  all_abr = np.concatenate(all_abr)
  all_ecog = np.concatenate(all_ecog)

  # Calculate the Pearson correlation using the "good" data.
  good = np.logical_and(np.logical_and(np.isfinite(all_abr),
                                       all_abr < max_spl),
                        np.logical_and(np.isfinite(all_ecog),
                                       all_ecog < max_spl))
  abr_thresh = all_abr[good]
  ecog_thresh = all_ecog[good]
  pearson_r = spstats.pearsonr(abr_thresh, ecog_thresh).statistic

  return all_abr, all_ecog, pearson_r


def plot_threshold_scatter(abr_thresh: np.ndarray, ecog_thresh: np.ndarray,
                           title: str = 'Comparson of Threshold',
                           axis_limit: float = 120,
                           color: str = 'b',
                           draw_unit: bool = True):
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
  plt.ylabel('ECochG (Channel 2) Threshold (dB)')
  if title:
    plt.title(title)
  axis_limit = min(min(np.max(abr_thresh), np.max(ecog_thresh)), axis_limit)
  plt.xlim(0, axis_limit)
  plt.ylim(0, axis_limit)
  if draw_unit:
    a = max(plt.axis())
    plt.plot([0, a], [0, a], '--')


def find_dprime(all_dprimes: Dict[str, DPrimeResult],
                spl: float) -> Tuple[np.ndarray, np.ndarray]:
  """Find the d' for all data in this dictionary of preparations at one SPL.
  Use the smooth d' data, which is best calculated with bilinear interpolaton.

  Args:
    all_dprimes: Dictionary mapping preparation name to d' results.
    spl: Which sound pressure level (SPL) to query the d'.  Must be one of the
      measured SPLs, as computed by the smooth polynomial.
  Returns:
    Matched ABR and ECoG arrays with the d' at the given SPL.  There should
    be three d's (one for each frequency) for each experiment.
  """
  abr_90s = []
  ecog_90s = []
  for k in all_dprimes:
    dp = all_dprimes[k]
    if (dp.cov_smooth_dprimes is None or spl not in dp.levels or
        dp.cov_smooth_dprimes.shape[2] < 2):
      continue
    index = dp.levels.index(spl)
    abr_90s.append(dp.cov_smooth_dprimes[:, index, 0])
    ecog_90s.append(dp.cov_smooth_dprimes[:, index, 1])

  return np.concatenate(abr_90s), np.concatenate(ecog_90s)

###############  Waveform and RMS Displays ######################


bad_animal_example = '20230720_control1_pre'   # No ABR or ECochG Response
good_animal_example = '20230720_control2_pre'  # No ABR/good ECochG Response


def extract_animal_names(exps: List[MouseExp]) -> List[str]:
  """From a list of mouse experiments, one trial per dataclass, extract the
  list of unique mouse names."""
  animal_names = []
  for exp in exps:
    k = exp.basename
    animal_name = k.split('-')[0]
    if animal_name not in animal_names:
      animal_names.append(animal_name)
  return animal_names


def filter_animals(exps: List[MouseExp],
                   animal_list: List[str]) -> List[MouseExp]:
  """Extract the MouseExp's for one or more specific mice."""
  filtered_exps = []
  for exp in exps:
    k = exp.basename
    if any([l in k for l in animal_list]):
      filtered_exps.append(exp)
  return filtered_exps


def filter_waveform_data(all_exps: List[MouseExp],
                         keep_list: Tuple[str] = ('control',),
                         drop_list: Tuple[str] = ('post', 'pre2', 'pre3',
                                                  'pre4', 'pre5', 'sricontrol',
                                                  'test', 'noacqfilter',
                                                  'Rearclosed'),
                         ) -> List[MouseExp]:
  """Filter a list of Mouse experiments, keeping the types of data we
  care about."""
  filtered_exps = []
  for exp in all_exps:
    k = exp.basename
    if any([l in k for l in drop_list]):
      continue
    if len(keep_list) and not any([l in k for l in keep_list]):
      continue
    filtered_exps.append(exp)
  return filtered_exps


GeorgeMouseDataDir = ('drive/Shareddrives/StanfordAudiology/'
                      'GeorgeMouseABR/CAP_ABR')


def XXcache_all_waveforms(base_dir: str = GeorgeMouseDataDir) -> None:
  """Read waveforms from disk, filter out the experiments we care about
  and store new abbreviated cache files.

  Args:
    base_dir: Where to look for the mouse data.
  """
  dirs = find_all_mouse_directories(base_dir)
  all_filtered_exps = []
  file_num = 0

  def cache_waveforms(file_num: int, all_filtered_exps: List[MouseExp]) -> int:
    if len(all_filtered_exps) > 0:
      good_waveform_cache = os.path.join(
        base_dir, f'good_waveform_cache{file_num:02d}.pkl')
      file_num += 1
      with open(good_waveform_cache, 'w') as f:
        print(f'Writing {len(all_filtered_exps)} waveforms to '
              f'{good_waveform_cache}')
        f.write(jsonpickle.encode(all_filtered_exps))
    return file_num

  for dir in dirs:
    cached_data = load_waveform_cache(dir)
    filtered_exps = filter_waveform_data(
      cached_data, keep_list=['control'],
      drop_list=['post', 'pre2', 'pre3', 'pre4', 'pre5', 'sricontrol',
                 'test', 'noacqfilter', 'Rearclosed'])
    del cached_data
    all_filtered_exps.extend(filtered_exps)
    print(f'Now have {len(all_filtered_exps)} trials.')
    if len(all_filtered_exps) > 500:
      file_num = cache_waveforms(file_num, all_filtered_exps)
      all_filtered_exps = []
  file_num = cache_waveforms(file_num, all_filtered_exps)  # Write remainder


def create_stack(exps: List[MouseExp]) -> np.ndarray:
  """Collect all the waveform data for a list of mouse experiments into a
    single tensor.

  Result is a 5d tensor:
    frequency, level, channel, time, trial
  """
  freqs = list(set([e.freq for e in exps]))
  levels = list(set([e.level for e in exps]))
  channels = list(set([e.channel for e in exps]))
  freqs.sort()
  levels.sort()
  channels.sort()
  data = np.zeros((len(freqs), len(levels), len(channels),
                   exps[0].single_trials.shape[0],
                   exps[0].single_trials.shape[1]))
  for i, exp in enumerate(exps):
    data[freqs.index(exp.freq), levels.index(exp.level),
         channels.index(exp.channel), :, :] = exp.single_trials
  return data, freqs, levels, channels


def show_mean_stack(stack, freq=1, channel=0, alpha=0.01):
  """Display a stack of mean ABR responses for different levels."""
  levels = list(range(0, stack.shape[1], 3))
  for i, level in enumerate(levels):
    plt.subplot(len(levels), 1, i+1)
    plt.plot(np.mean(stack[freq, level, channel, ...], axis=-1),
             alpha=alpha)
    if i != len(levels)-1:
      plt.gca().xaxis.set_tick_params(labelcolor='none')


def show_all_stack(stack: np.ndarray,
                   levels: Union[np.ndarray, List[float]],
                   freq: int = 1,
                   channel: int = 0, alpha: float = 0.01,
                   title: str = '',
                   skip_levels: int = 3,
                   relative_max: float = 1.5,
                   absolute_max: float = 0,
                   num_cols: int = 1,
                   col_num: int = 0) -> None:
  """Show a stack of all ABR waveforms across levels.  The number of plots is
  determined by the number of levels in stack, and skip_levels

  Args:
    stack: a 5 dimensional tensor from create_stack for one animal of shape
      frequency, level, channel, time, trial
    levels: which levels are defined in this stack
    freq: Desired frequency index (into stack) to plot
    channel: Desired channel index (into stack) to plot
    alpha: opaqueness of waveform plot.  Usually close to zero so we can
      overlap lots of waveforms
    title: What title to put on top of the waveform stack
    skip_levels: The increment (> 0) across levels for each subplot.
    relative_max: Limit y axis of plot to this factor of the max average
      waveform if the absolute value is not set.
    absolute_max: Limit y axis of plot to this absolute value.
    num_cols: How many columns of stacks to show
    col_num: which column to plot this time (0 <= col_num < num_cols)
  """
  levels2plot = levels[::skip_levels]
  t = np.arange(stack.shape[-2])/mouse_sample_rate
  for i, level in enumerate(levels2plot):
    plt.subplot(len(levels2plot), num_cols, i*num_cols+col_num+1)
    plt.plot(t*1000, stack[freq, levels.index(level), channel, ...],
             alpha=alpha)
    mean_stack = np.mean(stack[freq, levels.index(level), channel, ...],
                         axis=-1)
    plt.plot(t*1000, mean_stack, color='r')
    m = np.max(np.abs(mean_stack))
    if absolute_max > 0:
      plt.ylim(-absolute_max, absolute_max)
    else:
      plt.ylim(-relative_max*m, relative_max*m)
    wave_rms = np.sqrt(np.mean(stack[freq, levels.index(level),
                                     channel, ...]**2))
    plt.text(np.max(t*1000)*0.75, 1.20*m, f'Waveform RMS={wave_rms:5.3g}')
    ave_rms = np.sqrt(np.mean(mean_stack**2))
    plt.text(0.0, 1.20*m, f'Average RMS={ave_rms:5.3g}', color='red')

    plt.xlabel('Time (ms)')
    plt.ylabel(f'{level}dB')

    if i == 0:
      plt.title(title)
    if i != len(levels2plot)-1:
      plt.gca().xaxis.set_tick_params(labelcolor='none')


standard_freqs = [8000.0, 16000.0, 32000.0]
standard_levels = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
standard_channels = [1, 2]


def load_rms_data(
    base_dir: str = GeorgeMouseDataDir,
    cache_filename: str = 'good_waveform_cache.pkl') -> List[
      List[List[List[float]]]]:
  """Read in the cached RMS data from disk.

  The returned structure is a list (by frequency) of lists (by level) of lists
  (by channel) of results.
  """

  def total_size(all_rms):
    total = 0
    for fi, freq in enumerate(standard_freqs):
      for li, level in enumerate(standard_levels):
        for ci, channel in enumerate(standard_channels):
          total += len(all_rms[fi][li][ci])
    return total

  def concatenate(all_rms, new_rms):
    for fi, freq in enumerate(standard_freqs):
      for li, level in enumerate(standard_levels):
        for ci, channel in enumerate(standard_channels):
          all_rms[fi][li][ci].extend(new_rms[fi][li][ci])
    return all_rms

  rms_cache_file = os.path.join(base_dir, cache_filename)

  if not os.path.exists(rms_cache_file):
    # Run through all the waveform cache files (which are just the experiments
    # want to analyze) and accumulate the average ABR/ECochG response into a 3d
    # array of lists.
    global standard_freqs, standard_levels, standard_channels
    # Initialize all_rms as an empty list with the correct dimensions
    all_rms = [[[[] for _ in standard_channels]
                for _ in standard_levels]
               for _ in standard_freqs]
    for f in glob.glob(os.path.join(GeorgeMouseDataDir,
                                    'good_waveform_cache*.pkl')):
      with open(f, 'r') as f:
        exps = jsonpickle.decode(f.read())
        (new_rms, standard_freqs,
         standard_levels, standard_channels) = summarize_all_rms(exps, all_rms)
        print(f'Read {total_size(new_rms)} RMS results from {f}')
        del exps
        all_rms = concatenate(all_rms, new_rms)

    with open(rms_cache_file, 'w') as f:
      print(f'Writing RMS results to {rms_cache_file}')
      f.write(jsonpickle.encode(all_rms))
  else:
    with open(rms_cache_file, 'r') as f:
      all_rms = jsonpickle.decode(f.read())
      print(f'Read {total_size(all_rms)} RMS results from {rms_cache_file}')
  return all_rms


default_cache_name = 'good_waveform_cache.pkl'


def load_rms_data(
    base_dir: str = GeorgeMouseDataDir,
    cache_filename: str = default_cache_name) -> List[List[List[List[float]]]]:
  """Read in the cached RMS data from disk.

  The returned structure is a list (by frequency) of lists (by level) of lists
  (by channel) of results.
  """

  def total_size(all_rms):
    total = 0
    for fi, freq in enumerate(standard_freqs):
      for li, level in enumerate(standard_levels):
        for ci, channel in enumerate(standard_channels):
          total += len(all_rms[fi][li][ci])
    return total

  def concatenate(all_rms, new_rms):
    for fi, freq in enumerate(standard_freqs):
      for li, level in enumerate(standard_levels):
        for ci, channel in enumerate(standard_channels):
          all_rms[fi][li][ci].extend(new_rms[fi][li][ci])
    return all_rms

  rms_cache_file = os.path.join(base_dir, cache_filename)

  if not os.path.exists(rms_cache_file):
    # Run through all the waveform cache files (which are just the experiments
    # want to analyze) and accumulate the average ABR/ECochG response into a 3d
    # array of lists.
    global standard_freqs, standard_levels, standard_channels
    # Initialize all_rms as an empty list with the correct dimensions
    all_rms = [[[[] for _ in standard_channels]
                for _ in standard_levels]
               for _ in standard_freqs]
    for f in glob.glob(os.path.join(GeorgeMouseDataDir,
                                    'good_waveform_cache*.pkl')):
      with open(f, 'r') as f:
        exps = jsonpickle.decode(f.read())
        (new_rms, standard_freqs,
         standard_levels, standard_channels) = summarize_all_rms(exps, all_rms)
        print(f'Read {total_size(new_rms)} RMS results from {f}')
        del exps
        all_rms = concatenate(all_rms, new_rms)

    with open(rms_cache_file, 'w') as f:
      print(f'Writing RMS results to {rms_cache_file}')
      f.write(jsonpickle.encode(all_rms))
  else:
    with open(rms_cache_file, 'r') as f:
      all_rms = jsonpickle.decode(f.read())
      print(f'Read {total_size(all_rms)} RMS results from '
            f'{rms_cache_file}')
  return all_rms


def calculate_mean_std_rms_values(
    all_good_rms: List[List[List[List[float]]]]) -> Tuple[np.ndarray,
                                                          np.ndarray]:
  """Summarize the lists of lists of lists of lists of RMS values by
  calculating their mean and standard deviation.
  """
  global standard_freqs, standard_levels, standard_channels

  rms_means = np.zeros((len(standard_freqs), len(standard_levels),
                        len(standard_channels)))
  rms_stds = rms_means.copy()
  for fi, freq in enumerate(standard_freqs):
    for li, level in enumerate(standard_levels):
      for ci, channel in enumerate(standard_channels):
        rms_means[fi, li, ci] = np.mean(all_good_rms[fi][li][ci])
        rms_stds[fi, li, ci] = np.std(all_good_rms[fi][li][ci])
  return rms_means, rms_stds


###############  Analzing results re. number of trials ######################


def calculate_dprime_by_trial_count(filtered_abr_stack: np.ndarray,
                                    level_index: int = 9,
                                    noise_index: int = 0,
                                    freq_index: int = 1,
                                    channel_index: int = 1,
                                    min_count: int = 20,
                                    max_count: int = 20000,
                                    num_divisions: int = 10,
                                    ) -> Tuple[np.ndarray, np.ndarray,
                                               np.ndarray]:
  # The shape of the stacks array is Freqs x levels x channels x time x trials
  assert filtered_abr_stack.ndim == 5
  assert level_index < filtered_abr_stack.shape[1]
  assert noise_index < filtered_abr_stack.shape[1]
  assert freq_index < filtered_abr_stack.shape[0]
  assert channel_index < filtered_abr_stack.shape[2]

  # time_sample_count = filtered_abr_stack.shape[3]
  trial_count = filtered_abr_stack.shape[4]

  block_sizes = (trial_count / (2**np.arange(0, 
                                             num_divisions, 1.0))).astype(int)
  block_sizes = block_sizes[(block_sizes >= min_count) &
                            (block_sizes <= max_count)]
  dprime_mean_by_size = np.zeros(len(block_sizes))
  dprime_std_by_size = np.zeros(len(block_sizes))

  for i, block_size in enumerate(block_sizes):
    dps = []
    for block_start in range(0, trial_count - block_size + 1, block_size):
      block_end = block_start + block_size
      signal_data = filtered_abr_stack[freq_index, level_index,
                                       channel_index,
                                       :, block_start:block_end]
      noise_data = filtered_abr_stack[freq_index, noise_index,
                                      channel_index,
                                      :, block_start:block_end]
      dps.append(calculate_cov_dprime(signal_data, noise_data))
    dprime_mean_by_size[i] = np.mean(dps)
    dprime_std_by_size[i] = np.std(dps)
  return block_sizes, dprime_mean_by_size, dprime_std_by_size


def block_waveform_stack(filtered_abr_stack: np.ndarray,
                         block_size: int,
                         level_index = 9,
                         noise_index = 0,
                         freq_index = 1,
                         channel_index = 1,
                         repetition_count: int = 20,
                         ) -> Tuple[np.ndarray, np.ndarray]:
  """Yields blocks of random pieces from a waveform stack.  We choose trials
  at random, and the noise and signal data are independently sampled, both with
  replacement.

  Args:
    filtered_abr_stack: a num_freq x num_levels x num_channels x num_times x 
      num_trials array of preprocessed ABR recordings.
    block_size: Number of trials to include in the block
    level_index: Which signal level index to return
    noise_index: Which signal level index contains noise and no signal
    freq_index: Which stimulus frequency to return
    channel_index: Which recording channel to return
    repetition_count: How many block to return

  Returns:
    Tuple of signal and noise arrays, one block at a time.
  """
  assert filtered_abr_stack.ndim == 5
  assert level_index < filtered_abr_stack.shape[1]
  assert noise_index < filtered_abr_stack.shape[1]
  assert freq_index < filtered_abr_stack.shape[0]
  assert channel_index < filtered_abr_stack.shape[2]

  time_sample_count = filtered_abr_stack.shape[3]
  trial_count = filtered_abr_stack.shape[4]
  
  dps = []
  for j in range(repetition_count):
    # Note: transpose the resulting array slices because of this answer:
    #  https://stackoverflow.com/a/71489304
    signal_data = filtered_abr_stack[freq_index, level_index,
                                     channel_index, :,
                                     
                                     np.random.choice(trial_count,
                                                       block_size)].T
    noise_data = filtered_abr_stack[freq_index, noise_index,
                                    channel_index, :,
                                    np.random.choice(trial_count,
                                                     block_size)].T
    yield signal_data, noise_data


def create_synthetic_stack(noise_level=1,
                           num_times=1952,
                           num_trials=1026):
  """Create a synthetic stack of ABR recordings so we can investigate d'
  behaviour for really large number of trials.
  """
  t = np.arange(num_times)/1000
  order = 4
  b = 1
  f = 5
  gammatone = 1000*t**(order-1)*np.exp(-2*np.pi*b*t)*np.cos(2*np.pi*f*t)
  # gammatone = np.cos(2*np.pi*f*t)*np.hamming(num_times)
  plt.plot(t, gammatone)

  # The shape of the stacks array is Freqs x levels x channels x time x trials
  stack = noise_level*np.random.normal(size=(1, 2, 1, num_times, num_trials))
  stack[0, 1, 0, :, :] += np.expand_dims(gammatone, axis=[1])
  return stack


def snr_vs_window_size(abr_stack: np.ndarray,
                       channel_index: int = 0,  # 0 is ABR, 1 is ECochG
                       freq_index: int = 1,
                       window_step: int = 50, 
                       ) -> Tuple[np.ndarray, 
                                                            np.ndarray]:
  time_sample_count = abr_stack.shape[3]
  trial_count = abr_stack.shape[4]

  level_index = abr_stack.shape[1]-1
  noise_index = 0

  time_windows = np.arange(0, time_sample_count, window_step)
  snrs = np.zeros((len(time_windows), len(time_windows))) * np.nan
  for i, time_start in enumerate(time_windows):
    for j, time_end in enumerate(time_windows):
      if time_start >= time_end:
        continue
      time_window = np.arange(time_start, time_end)
      signal_data = abr_stack[freq_index, level_index,
                              channel_index, time_window, :]
      noise_data = abr_stack[freq_index, noise_index,
                             channel_index, time_window, :]
      signal_rms = np.sqrt(np.mean(np.mean(signal_data, axis=-1)**2, axis=-1))
      # Shuffle the noise data to make sure it has no signal.
      noise_data = noise_data.T
      np.random.shuffle(noise_data)
      noise_data = noise_data.T
      noise_rms = np.sqrt(np.mean(np.mean(noise_data, axis=-1)**2, axis=-1))
      snr = signal_rms/noise_rms
      snrs[i, j] = snr  # signal_rms
  return snrs, time_windows


def stack_t_test(filtered_abr_stack: np.ndarray,
                 channel_index:int = 1,  # ABR
                 freq_index:int = 1,  # 160000
                 plot_pvals: bool = True,
                 ) -> Tuple[np.ndarray,
                                                          List[int]]:
  """Compute the average signal and noise response (averaging over trials),
  as a function of the bnlock size (non overlapping blocks, no bootstrapping).
  Then form the distribution of average signal ABR and noise ABRs in order to
  calculate the student t-test, and plot the p-values as a function of signal
  level and block size.

  Args:
    filtered_abr_stack: recorded waveform tensor of shape:
      num_freqs x num_levels x num_channels x num_times x num_trials
  """
  trial_count = filtered_abr_stack.shape[-1]
  num_divisions = 10
  min_count = 20
  max_count = 20000

  block_sizes = (trial_count / (2**np.arange(0,
                                            num_divisions, 1.0))).astype(int)
  block_sizes = block_sizes[(block_sizes >= min_count) &
                            (block_sizes <= max_count)]

  pvals = np.zeros((filtered_abr_stack.shape[1], len(block_sizes)))
  for level_index in range(filtered_abr_stack.shape[1]):
    t_stats = []
    for j, block_size in enumerate(block_sizes):
      noise_response = []
      abr_response = []
      for signal, noise in block_waveform_stack(filtered_abr_stack,
                                                level_index=level_index,
                                                freq_index=freq_index,
                                                channel_index=channel_index,
                                                block_size=block_size):
        abr_response.append(np.sqrt(np.mean(np.mean(signal, axis=-1)**2)))
        noise_response.append(np.sqrt(np.mean(np.mean(noise, axis=-1)**2)))
      t_stat = spstats.ttest_ind(abr_response, noise_response)
      t_stats.append(t_stat)
      pvals[level_index, j] = t_stat.pvalue
    if plot_pvals:
      plt.semilogy(block_sizes, [t.pvalue for t in t_stats], 
                  label=f'Signal Level {10*level_index}');
    if level_index == 0:
      print('p values for signal level 0:', [float(t.pvalue) for t in t_stats])
  if plot_pvals:
    plt.legend()
    plt.xlabel('Block Size (trials)')
    plt.ylabel('p-value')
    plt.title('p-value vs. Block Size');
  return pvals, block_sizes


if False:
  synthetic_stack = create_synthetic_stack(noise_level=10, num_trials=16384)
  plt.title('Synthetic ABR Waveform')

  (block_sizes, dprime_mean_by_size,
   dprime_std_by_size) = calculate_dprime_by_trial_count(synthetic_stack,
                                                         signal_index=1,
                                                         noise_index=0,
                                                         freq_index=0,
                                                         channel_index=0,)

  plt.errorbar(block_sizes, dprime_mean_by_size, dprime_std_by_size)
  plt.xscale('log')
  plt.xlabel('Block Size (trials)')
  plt.ylabel("d' Estimate")
  plt.title("Synthetic d' vs. Trial Count")

###############  Main program, so we can run this offline ####################

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'waveforms', ('waveforms', 'dprimes', 'check'),
                  'Which processing to do on this basedir.')
flags.DEFINE_string('basedir',
                    'drive/Shareddrives/StanfordAudiology/'
                    'GeorgeMouseABR/CAP_ABR',
                    'Base directory to find the ABRPresto mouse data')
flags.DEFINE_string('waveforms_cache', mouse_waveforms_pickle_name,
                    'Where to cache all the waveforms in this directory')
flags.DEFINE_string('dprimes_cache', mouse_dprimes_pickle_name,
                    'Where to cache the dprimes in this directory')
flags.DEFINE_string('filter', '',
                    'Which subdirectories to process, ignore rest.')
flags.DEFINE_integer('max_cache_gbytes', 10,
                     'Maximum size of one cache file (GBytes).')
flags.DEFINE_integer('first_sample', 0,
                     'Start sample # of the temporal window to extract from '
                     'each ABR waveform')
flags.DEFINE_integer('last_sample', -1,
                     'End sample # of the temporal window to extract from a '
                     'waveform, (including last sample is indicated with -1)')


def waveform_caches_present(dir: str, waveform_pickle_name: str) -> int:
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


def cache_waveform_one_dir(dir: str, waveform_pickle_name: str,
                           max_files: int = 0,
                           max_bytes: float = 10e9) -> None:
  """Read all the CSV files and convert them into pickled numpy arrays.  CSV
  files take a long time to read and parse, so this is an important speedup.

  Args:
    dir: Top level directory to find all the waveform CSV files.
    waveform_pickle_name: Basic pickle file name, of the form  x.pkl.  We
      remove the .pkl name and look for files of the form x*.pkl, as the
      pickle files are big and we had to split them into multiple pieces,
      indexed by a number.
    max_files: For debugging.  Limit the number of CSV files we read.
    max_bytes: Maximum number of bytes, or thereabouts, to put in each pickle
      file.

  """
  num_good = waveform_caches_present(dir, waveform_pickle_name)
  if num_good:
    print(f'Skipping waveforms and dprimes in {dir} because they are '
          f'{num_good} cached files.')
    return
  print(f'Processing up to {max_files} files or {max_bytes/1e9} GB of '
        f'CSV waveforms in {dir}')
  cache_all_mouse_dir(dir, waveform_pickle_name, debug=True,
                      max_files=max_files, max_bytes=max_bytes)


def cache_dprime_one_dir(dir: str,
                         waveform_cache_name: str, dprime_cache_name: str,
                         first_sample: int = 0,
                         last_sample: int = 0,
                         ) -> None:
  if os.path.exists(os.path.join(dir, dprime_cache_name)):
    print(f'Cache directory exists for {dir}')
    return
  print(f'Loading waveforms from {dir} to compute d\'s.')
  all_exps = load_waveform_cache(dir, waveform_cache_name)
  if all_exps:
    dprimes = calculate_all_summaries(all_exps)
    cache_dprime_data(dir, dprimes, dprime_cache_name)
  else:
    print('  No waveform data to process for dprimes.')


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
        cache_dprime_one_dir(dir, FLAGS.waveforms_cache, FLAGS.dprimes_cache,
                             FLAGS.first_sample, FLAGS.last_sample)
  else:
    print(f'Unknown processing mode: {FLAGS.mode}')


if __name__ == '__main__':
  app.run(main)
