import datetime
import os

from absl import app, flags
import numpy as np
from numpy.typing import ArrayLike, NDArray

import matplotlib.pyplot as plt

from typing import Dict, Optional, Union

# Note this syntax (import *) is in general a bad idea.  But it's a good
# solution in colab so you can easily redefine a function that was already
# defined in the abr.py file.  Just don't forget to check in the new version
# at some point.
#   https://www.geeksforgeeks.org/why-import-star-in-python-is-a-bad-idea/

from abr_metrics import *
from abr_george import BilinearInterpolation, PositivePolynomial

# We save the raw data with Pickle because raw JSON does not support Numpy
#  https://jsonpickle.github.io/
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

DistributionArray = NDArray

# DistributionList with num_block_counts 3D arrays.  Each array of size
#       num_levels x bootstrap_repetitions x trial_count
# for one metric type
DistributionList = List[DistributionArray]

# MultipleDistribution Lists are combined into a dictionary, keyed by the 
# metric name.
MetricDistributionDict = Dict[str, List[NDArray]]

DPrimeDict = Dict[str, NDArray]

######################  Cache Partial Results #############################

Synthetic_ABR_Cache_Dir: str = '/tmp'
Synthetic_ABR_Cache_Force: bool = False

def cache_exists(cache_file: str) -> bool: 
  if Synthetic_ABR_Cache_Force:
     return False
  return os.path.exists(os.path.join(Synthetic_ABR_Cache_Dir, cache_file))

def save_to_cache(data: Dict[str, NDArray], cache_file: str):
  data['date'] = datetime.datetime.now()
  pickle_file = os.path.join(Synthetic_ABR_Cache_Dir, cache_file)
  with open(pickle_file, "w") as f:
    f.write(jsonpickle.encode(data))
    print(f'  Cached data for {list(data.keys())} data '
          f'into {pickle_file}.')

def restore_from_cache(pickle_filename: str) -> Dict[str, NDArray]:
  pickle_file = os.path.join(Synthetic_ABR_Cache_Dir, pickle_filename)
  if os.path.exists(pickle_file):
      with open(pickle_file, "r") as f:
          data = jsonpickle.decode(f.read())
          print(f'  Read cached data from {pickle_file}.')
      return data
  return {}

######################  Create Experimental Data  #############################

def create_exp_stack(signal_levels: List[float] = [],
                     num_trials: int = 4096, 
                     num_times: int = 1952,
                     noise_level: float = 1.0,
                     cache_file: str = 'exp_stack.pkl') -> DistributionArray:
  """Compute a stack of simulated ABR data using a Gammatone model.
   
  Returns: 
    A NDArray of size num_levels x num_times x num_trials
  """
  if cache_exists(cache_file):
    data = restore_from_cache(cache_file)
    return data['signal_levels'], data['waveforms']

  if not len(signal_levels):
    signal_levels = np.asarray([0, .01, .02, .03, .04, .05, .06, .07, .08, .09,
                                .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
  exp_stack = create_synthetic_stack(noise_level=noise_level,
                                     signal_levels=signal_levels,
                                     num_times=num_times,
                                     num_trials=num_trials)
  # exp_stack.shape is Number of levels, # of time samples, # of trials
  save_to_cache({'signal_levels': signal_levels,
                 'noise_level': noise_level,
                 'waveforms': exp_stack}, cache_file)
  return signal_levels, exp_stack


def plot_exp_stack_waveform(
    exp_stack: NDArray,  # Shape: num_levels x num_times x num_trials
    level_index: int = -1, 
    plot_file: Optional[str] ='WaveformStackDisplay.png'): 
  assert exp_stack.ndim == 3, f'Expected three dimensions in exp_stack, got {exp_stack.shape}'
  plt.clf()
  plt.plot(exp_stack[level_index, :, 0], label='One trial')
  plt.plot(np.mean(exp_stack[-1, ...], axis=level_index), label='Average')
  plt.title('ABR Response at Highest Sound Level')
  plt.legend()
  if plot_file:
    plt.savefig(plot_file)

def plot_peak_illustration(exp_stack: NDArray, # Shape: num_levels x num_times x num_trials
                           level_index: int = -1,
                           plot_file: Optional[str] ='WaveformPeakIllustration.png'):
  assert exp_stack.ndim == 3, f'Expected three dimensions in exp_stack, got {exp_stack.shape}'
  mean_signal_response = np.mean(exp_stack[-1, ...], axis=1)
  peak_index = np.argmax(mean_signal_response)

  plt.clf()
  plt.subplot(2,1,1);
  plt.plot(np.mean(exp_stack[-1, ...], axis=1))
  plt.plot(peak_index, mean_signal_response[peak_index], 'ro')
  plt.subplot(2,1,2);
  plt.plot(np.mean(exp_stack[-1, ...], axis=1))
  plt.plot(peak_index, mean_signal_response[peak_index], 'ro')
  plt.xlim(400, 500)
  if plot_file:
    plt.savefig(plot_file)


def plot_peak_metric (exp_stack: NDArray, level_index: int = -1,
                         plot_file: Optional[str] ='WaveformPeakMetric.png'):
  assert exp_stack.ndim == 3, f'Expected three dimensions in exp_stack, got {exp_stack.shape}'
  mean_response = np.mean(exp_stack[-1, ...], axis=1)
  peak_index = np.argmax(mean_response)
  noise = exp_stack[0, :, 0]
  plt.clf()
  plt.plot(noise, label='Noise Response')
  plt.plot(mean_response, label='Mean Signal')
  plt.axhline(np.std(noise), color='r', linestyle=':', label='Noise RMS')
  plt.axhline(np.max(mean_response), color='g', linestyle=':', label='Signal Peak')
  plt.legend()
  peak_measure = PeakMetric().compute(exp_stack[-1, ...])
  plt.title(f'Peak Amplitude to RMS Noise is {peak_measure}')
  if plot_file:
    plt.savefig(plot_file)


def plot_baselines(exp_stack: NDArray,
                   stack_signal_levels: ArrayLike, 
                   plot_file: Optional[str] = 'BaselineXXX.png') -> None:
  assert exp_stack.ndim == 3, f'Expected three dimensions in exp_stack, got {exp_stack.shape}'
  num_levels, num_times, num_trials = exp_stack.shape

  # Peak Baseline Plot
  plt.clf()
  signal_level = 8
  mouse_sample_rate = 24414 * 8  # From George's Exp Notes, 8x oversampling
  window_start: int = int(1.75e-3*mouse_sample_rate),
  window_end: int = int(3e-3*mouse_sample_rate)
  signal_average = np.mean(exp_stack[signal_level, :, :], axis=1)
  peak_index = np.argmax(np.abs(signal_average))
  noise_average = np.mean(exp_stack[0, :, :], axis=1)
  plt.clf()
  plt.plot(noise_average, label='Average Noise Response')
  plt.plot(signal_average, label='Average Signal Response')

  plt.axhline(np.std(noise_average), color='k', linestyle=':', label='Noise RMS')
  plt.axvline(window_start, color='g', linestyle='--')
  plt.axvline(window_end, color='g', linestyle='--')

  plt.plot(peak_index, signal_average[peak_index], 'ro', label='Signal Abs Peak')
  plt.legend()
  plt.xlabel('Sample #')
  peak_measure = PeakMetric().compute(exp_stack[signal_level, ...])
  plt.title(f'Peak Amplitude to RMS Noise is {peak_measure[0]:.2f}');##################  Compute a distribution and plot it  #######################
  if plot_file:
    plt.savefig(plot_file.replace('XXX', 'Peak'))

  # Per Trial RMS
  plt.clf()
  signal_levels = [np.std(np.mean(exp_stack[i, :, :], axis=1)) for i in range(num_levels)]
  signal_noise_levels = [np.std(exp_stack[i, :, :]) for i in range(num_levels)]
  noise_level = np.std(exp_stack[0, :, :])

  plt.plot(stack_signal_levels, signal_noise_levels, label='Signal + Noise RMS')
  plt.plot(stack_signal_levels, noise_level*np.ones(stack_signal_levels.shape), label='Noise RMS')
  plt.plot(stack_signal_levels, signal_levels, label='Signal RMS')
  plt.legend();
  plt.xlabel('Signal Level')
  plt.ylabel('RMS')
  plt.title('Per Trial RMS Energy of Signal and Noise');
  if plot_file:
    plt.savefig(plot_file.replace('XXX', 'PerTrialRMS'))

  # Total RMS Baseline Plot
  trial_counts = [100, 200, 400, 800, 1600, 3200, 4096]
  signal_levels = [np.std(np.mean(exp_stack[signal_level, :, :tc], axis=1)) for tc in trial_counts]
  noise_levels = [np.std(np.mean(exp_stack[0, :, :tc], axis=1)) for tc in trial_counts]
  # noise_level = np.std(np.mean(exp_stack[0, :, :], axis=1))

  plt.clf()
  plt.semilogx(trial_counts, noise_levels, label='Noise RMS')
  # plt.plot(stack_signal_levels, noise_level*np.ones(stack_signal_levels.shape), label='Noise RMS')
  plt.semilogx(trial_counts, signal_levels, label='Signal RMS')
  plt.legend();
  plt.xlabel('Number of Trials')
  plt.ylabel('RMS')
  plt.title('Total RMS Energy of Signal and Noise');
  if plot_file:
    plt.savefig(plot_file.replace('XXX', 'TotalRMS'))


def plot_distribution_histogram_comparison(
    top_dist: NDArray, bottom_dist: NDArray, 
    top_label: str = 'Top', bottom_label: str = 'Bottom', 
    bin_count: int = 20, 
    plot_file: Optional[str] = 'HistogramComparison_top_bottom.png',
                                           ) -> None:
  # Original shaped: num_levels x bootstrap_repetitions x trial_count
  assert top_dist.ndim == 3, f'Wanted three dimensions on top, got {top_dist.shape}'
  assert bottom_dist.ndim == 3, f'Wanted three dimensions on bottom, got {bottom_dist.shape}'
  top_dist = np.reshape(top_dist, (top_dist.shape[0], -1))
  bottom_dist = np.reshape(bottom_dist, (bottom_dist.shape[0], -1))

  plt.clf()
  plt.subplot(2, 1, 1)
  plt.hist(top_dist[0, ...], bin_count, label=f'Noise ({top_label})')
  plt.hist(top_dist[-1, ...], bin_count, label=f'Signal ({top_label})')
  plt.legend();

  plt.subplot(2, 1, 2)
  plt.hist(bottom_dist[0, ...], bin_count, label=f'Noise ({bottom_label})')
  plt.hist(bottom_dist[-1, ...], bin_count, label=f'Signal ({bottom_label})')

  plt.legend();

  if plot_file:
    plot_file = plot_file.replace('top', top_label)
    plot_file = plot_file.replace('bottom', bottom_label)
    plt.savefig(plot_file)

##################  Compute a distribution and plot it  #######################

distribution_names = {
   'TotalRMS': TotalRMSMetric,
   'TrialRMS': TrialRMSMetric,
   'Covariance': CovarianceMetric,
   'Peak': PeakMetric,
   # 'Presto': PrestoMetric # always has 500 splits!
}

def compute_all_distributions(exp_stack: DistributionArray, 
                              block_sizes: List[int],
                              bootstrap_repetitions: int = 30
                              ) -> Tuple[MetricDistributionDict,
                                                    List[int]]:
  distributions = {}
  for distribution_name in distribution_names:
    cache_file = f'Distribution_{distribution_name}_Cache.pkl'
    if cache_exists(cache_file):
      data = restore_from_cache(cache_file)
      distribution = data['distribution']
      block_sizes = data['block_sizes']
      distributions[distribution_name] = distribution
    else:
      metric:Metric = distribution_names[distribution_name]()
      (distribution, 
       block_sizes) = metric.compute_distribution_by_trial_size(
         exp_stack, block_sizes, bootstrap_repetitions=bootstrap_repetitions)
      assert distribution[0].ndim == 3, f'Expected three dimensions in distribution[0], got {distribution[0].shape}'
      distributions[distribution_name] = distribution
      save_to_cache({'distribution': distribution,
                     'bootstrap_repetitions': bootstrap_repetitions,
                     'block_sizes': block_sizes}, 
                    cache_file)
  return distributions, block_sizes

def plot_distribution_vs_trials(
    distribution_list: DistributionList, 
    block_sizes: List[int] = [], 
    signal_levels: List[float] = [],
    levels_to_display: List[int] = [],
    plot_file: str = 'Distribution_vs_number_of_trials.png'):
  """Just for the peak metric..."""
  # num_levels x bootstrap_repetitions x trial_count
  assert isinstance(distribution_list, list), f'Expected a list of arrays, got {type(distribution_list)}'
  assert distribution_list[0].ndim == 3, f'Expected three dimensions in distribution, got {distribution_list.shape}'

  # Calculate stats over bootstrap samples and over each trial in sample!!!
  means = [np.expand_dims(np.mean(d, axis=(1,2)), 1) for d in distribution_list]
  stds = [np.expand_dims(np.std(d, axis=(1,2)), 1)  for d in distribution_list]
  means = np.concatenate(means, axis=1)
  stds = np.concatenate(stds, axis=1)
  if len(signal_levels) != means.shape[0]:
    signal_levels = range(means.shape[0])
  plt.clf()
  if len(levels_to_display) == 0:
    levels_to_display = range(means.shape[0])
  for i in levels_to_display:
    plt.errorbar(block_sizes, means[i, :], yerr=stds[i, :], 
                 label=signal_levels[i])
  plt.xlabel('Number of Trials')
  plt.ylabel('Peak to Noise RMS Value')
  plt.gca().set_yscale('log')
  plt.gca().set_xscale('log')
  plt.axhline(3, ls=':')
  plt.title(f'Distribution vs. Number of Trials ')
  plt.legend()
  plt.savefig(plot_file)


def plot_distribution_analysis(dist_list: DistributionList, 
                               block_sizes: List[int], 
                               ylabel: str = 'Distribution of the Covariance',
                               plot_file: str = 'DistributionAnalysis.png'):
  # distribution is List with num_block_counts 3D arrays.  Each array of size
  #       num_levels x bootstrap_repetitions x trial_count
  plt.figure(figsize=(8, 6))

  if dist_list[0].shape[-1] > 1:
    means = [np.mean(dist_list[b][-1, ...]) for b in range(len(block_sizes))]
    stds = [np.std(dist_list[b][-1, ...]) for b in range(len(block_sizes))]
  else:
    means = [np.mean(dist_list[b][-1, ...]) for b in range(len(block_sizes))]
    stds =[np.ones(means[b].shape) for b in range(len(block_sizes))]
  plt.subplot(2, 2, 1)
  plt.errorbar(block_sizes, means, yerr=4*np.asarray(stds),
          label='Signal')
  meann = [np.mean(dist_list[b][0, ...]) for b in range(len(block_sizes))]
  stdn = [np.std(dist_list[b][0, ...]) for b in range(len(block_sizes))]
  plt.errorbar(block_sizes, meann, yerr=4*np.asarray(stdn),
          label='Noise')
  plt.gca().set_xscale('log')
  plt.legend()
  # plt.xlabel('Trial Count')
  plt.ylabel(ylabel)
  plt.title(r'Distributions - Level 9 (errorbars are $4\sigma$)');

  plt.subplot(2, 2, 2)
  dprime = ((np.asarray(means) - np.asarray(meann)) /
            (np.sqrt((np.asarray(stds)**2 + np.asarray(stdn)**2)/2)))
  if 'Peak' in ylabel or 'Total RMS' in ylabel:
    # Last point has zero variance because there is no change across bootstraps
    # because each sample returns *all* the same points.
    plt.plot(block_sizes[1:], dprime[1:])
  else:
    plt.plot(block_sizes, dprime)
  plt.gca().set_xscale('log')
  # if np.max(dprime) > 100:
  #   plt.gca().set_yscale('log')
  plt.ylabel('d\'');
  # Conclusion: d' grows because noise covariance distributions gets closer to 0.

  plt.subplot(2, 2, 3)
  plt.plot(block_sizes, means, label='Signal')
  plt.plot(block_sizes, meann, label='Noise')
  plt.gca().set_xscale('log')
  plt.xlabel('Trial Count')
  plt.ylabel(ylabel)
  plt.legend()
  plt.title('Means of Distributions')
  # plt.xlabel('Trial Count');

  plt.subplot(2, 2, 4)
  plt.plot(block_sizes, stds, label='Signal')
  plt.plot(block_sizes, stdn, label='Noise')
  plt.gca().set_xscale('log')
  plt.legend()
  plt.title(r'$\sigma$ of Distribution')
  plt.xlabel('Trial Count');
  plt.xlabel('$\\sigma$');

  plt.subplots_adjust(wspace=0.3, hspace=0.4);

  if plot_file:
    plt.savefig(plot_file)

##################   D' Analysis  #######################

def calculate_all_dprime(
    distribution_dict: MetricDistributionDict,
    block_sizes: List[int],
    cache_file: str = 'all_dprimes.pkl') -> DPrimeDict:
  """
  Returns
    Dictionary of d' arrays, each array of size 
      num_trial_sizes x num_levels x num_trials
  """
  if cache_exists(cache_file):
    data = restore_from_cache(cache_file)
    return data['dprime_dict']

  dprime_dict = {}
  for distribution_name in distribution_dict:
    distribution_list = distribution_dict[distribution_name]
    dprimes = None
    # block_sizes = []  # Recompute from the data
    for i in range(len(distribution_list)):  # Over block sizes
      distribution = distribution_list[i]
      # block_sizes.append(distribution.shape[-1])
      assert distribution.ndim == 3, f'Expected three dimensions in distribution, got {distribution.shape}'
      num_levels, num_bootstraps, num_trials = distribution.shape
      if dprimes is None:
        # Want num_trial_sizes x num_levels x num_trials
        dprimes = np.zeros((len(distribution_list), num_levels, num_bootstraps))
      for j in range(1, num_levels):
        for k in range(num_bootstraps):
          if distribution_name == 'Peak':
            # With peak metric there is no distribution, just return the mean
            dprimes[i, j, k] = np.mean(distribution[j, k, :])
          else:
            dprimes[i, j, k] = calculate_dprime(distribution[j, k, :], 
                                                distribution[0, k, :])
    dprime_dict[distribution_name] = dprimes
    dprimes = None

  save_to_cache({'dprime_dict': dprime_dict,
                 'block_sizes': block_sizes}, cache_file)
  return dprime_dict

def plot_dprime_vs_trials(dprimes, name='', block_sizes: List[int] = [],
                          sound_levels:List[float] = [],
                          sound_levels_to_plot: List[int] = [],
                          ylabel='d\'',
                          plot_file: str = 'dprime.png'):
  # Expect num_trial_sizes x num_levels x num_trials
  plt.clf()
  dprimes = np.mean(dprimes, axis=2)  # Now num_trial_sizes x num_levels
  if len(sound_levels) == 0:
    sound_levels = range(dprimes.shape[1])
  if len(sound_levels_to_plot) == 0:
    sound_levels_to_plot = range(dprimes.shape[1])
  if len(block_sizes):
    plt.semilogx(block_sizes, dprimes[:, sound_levels_to_plot])
  else:
    plt.plot(dprimes[:, sound_levels_to_plot])
  plt.xlabel('Number of Trials')
  plt.ylabel(ylabel)
  plt.legend([f'Level={l}' for l in sound_levels[sound_levels_to_plot]])
  plt.title(name)
  if plot_file:
    plt.savefig(plot_file)

def plot_dprimes_vs_sound_level_distribution(
      dprimes: NDArray, trial_counts: List[int], signal_levels: ArrayLike, 
      ylabel: str = 'd\'',
      plot_file: str = 'DPrimeVsLevel.png'):
  # Dprimes is Expect num_trial_sizes x num_levels x num_trials
  dp_mean = np.mean(dprimes, axis=2)
  dp_std = np.std(dprimes, axis=2)
  plt.clf()
  for i in reversed(range(0, len(trial_counts), 2)):
    plt.errorbar(signal_levels, dp_mean[i, :], capsize=5,
                yerr=dp_std[i, :], label=f'Trial Count={trial_counts[i]}');
  plt.legend();
  plt.xlabel('Sound Level');
  plt.ylabel(ylabel)
  plt.title('Metric vs. Amplitude and Number of Trials');
  if plot_file:
    plt.savefig(plot_file)

def plot_dprimes_vs_sound_level(
    dprime_dict: Dict[str, NDArray], 
    signal_levels: ArrayLike,
    first_metric: str = 'Covariance',
    second_metric: str = 'TotalRMS',
    third_metric: str = 'Peak',
    plot_file: str = 'DistributionByLevel_first_second_third.png') -> None:
  first_dprimes = dprime_dict[first_metric]
  second_dprimes = dprime_dict[second_metric]
  third_dprimes = dprime_dict[third_metric]
  plt.clf()
  plt.plot(signal_levels, np.nanmean(first_dprimes[0, :, :], axis=-1), 
           label=first_metric)
  plt.plot(signal_levels, np.nanmean(second_dprimes[0, :, :], axis=-1), 
           label=second_metric)
  plt.plot(signal_levels, np.nanmean(third_dprimes[0, :, :], axis=-1), 
           label=third_metric)
  plt.xlabel('Sound Level (linear a.u.)')
  plt.ylabel('d\'')
  plt.title('Distribution Separation vs. Sound Level')
  plt.legend();
  plot_file = plot_file.replace('first', first_metric)
  plot_file = plot_file.replace('second', second_metric)
  plot_file = plot_file.replace('third', third_metric)
  plt.savefig(plot_file)

def plot_dprimes_vs_trials(
    dprime_dict: Dict[str, NDArray], 
    first_metric: str = 'Covariance',
    second_metric: str = 'TotalRMS',
    third_metric: str = 'Peak',
    level_num: int = -1, block_sizes: List[int] = [], 
    plot_file: str = 'DistributionByTrials_first_second_third.png'):
  # Expect num_trial_sizes x num_levels x num_trials
  first_dprimes = dprime_dict[first_metric]
  assert first_dprimes.ndim == 3, f'Expected three dimensions in dprimes, got {first_dprimes.shape}'
  second_dprimes = dprime_dict[second_metric]
  assert second_dprimes.ndim == 3, f'Expected three dimensions in dprimes, got {second_dprimes.shape}'
  third_dprimes = dprime_dict[third_metric]
  assert third_dprimes.ndim == 3, f'Expected three dimensions in dprimes, got {third_dprimes.shape}'
  plt.clf()
  plt.semilogx(block_sizes, np.mean(first_dprimes[:, level_num, :], axis=-1), label=first_metric)
  plt.semilogx(block_sizes, np.mean(second_dprimes[:, level_num, :], axis=-1), label=second_metric)
  plt.semilogx(block_sizes, np.mean(third_dprimes[:, level_num, :], axis=-1), label=third_metric)
  plt.xlabel('Number of Trials')
  plt.ylabel('d\'')
  plt.title(f'd\' Separation vs. Number of Trials (level {level_num})')
  plt.legend()
  plot_file = plot_file.replace('first', first_metric)
  plot_file = plot_file.replace('second', second_metric)
  plot_file = plot_file.replace('third', third_metric)
  plt.savefig(plot_file)

##################   Thresholds  #######################

def compute_thresholds(data: List[NDArray],  # Usually d' data
                       signal_levels: List[float], 
                       trial_counts: List[int],
                       metric_name: str,
                       thresholds: List[float] = [1.0, 2.0, 3.0,],
                       plot_file: str = 'ThresholdVsTrials_metric.png'):
  # Expect num_trial_sizes x num_levels x num_trials
  assert data.ndim == 3, f'Expected three dimensions in data, got {data.shape}'
  data = np.mean(data, axis=2)  # Average over trials, now size x levels

  # pp = PositivePolynomial(semilogx=True)
  pp = BilinearInterpolation()
  plt.clf()
  for desired_threshold in thresholds:
    threshold = np.zeros(data.shape[0]) # Number of Trials
    for i in range(data.shape[0]):
      pp.fit(signal_levels, data[i, :])
      threshold[i] = pp.threshold(desired_threshold)
    plt.semilogx(trial_counts, threshold, 
                 label=f'Decision Criteria = {desired_threshold}')
  plt.legend()
  plt.xlabel('Number of trials');
  plt.ylabel('Amplitude of Signal for Decision (a.u.)')
  plt.title(f'{metric_name}: Sound Level Threshold vs. Decision Criteria');
  plot_file = plot_file.replace('metric', metric_name)
  if plot_file:
    plt.savefig(plot_file)

##################   Main Program  #######################

FLAGS = flags.FLAGS
flags.DEFINE_integer('max_trials', 4096, 'Maximum number of trials to precompute', lower_bound=100)
flags.DEFINE_integer('num_times', 1952, 'Number of time samples to compute', lower_bound=100)
flags.DEFINE_integer('num_bootstraps', 30, 'How many bootstraps to use when computing statistics', lower_bound=10)
flags.DEFINE_float('noise_level', 1.0, 'What noise level to use throughout these experiments')
# flags.DEFINE_string('cache_dir', '/tmp', 'Where to cache the precomputed data')

def main(*argv):
  global Synthetic_ABR_Cache_Dir
  Synthetic_ABR_Cache_Dir = FLAGS.cache_dir

  stack_signal_levels, exp_stack = create_exp_stack(noise_level=FLAGS.noise_level,
                                                    num_times=FLAGS.num_times,
                                                    num_trials=FLAGS.max_trials)
  num_levels, num_times, num_trials = exp_stack.shape
  plot_exp_stack_waveform(exp_stack)
  plot_peak_illustration(exp_stack)
  plot_peak_metric(exp_stack)
  plot_baselines(exp_stack, stack_signal_levels)

  # Now pre calculate all the metrics.
  metric_total_rms = TotalRMSMetric()
  dist_total_rms = metric_total_rms.compute_distribution(exp_stack)

  metric_cov = CovarianceMetric()
  dist_cov = metric_cov.compute_distribution(exp_stack)

  metric_peak = PeakMetric()
  dist_peak = metric_peak.compute_distribution(exp_stack)


  num_divisions = 14
  block_sizes = (num_trials / (2 ** np.arange(0, 
                                              num_divisions, 
                                              1.0))).astype(int)

  distribution_dict, block_sizes = compute_all_distributions(
    exp_stack, block_sizes, bootstrap_repetitions=FLAGS.num_bootstraps)
  plot_distribution_histogram_comparison(
    distribution_dict['Covariance'][-1], distribution_dict['Covariance'][0], 
    top_label=f'Covariance: trial count={block_sizes[-1]}', 
    bottom_label=f'Covariance: trial count={block_sizes[0]}')

  # Just for the Peak distribution
  plot_distribution_vs_trials(
    distribution_dict['Peak'], block_sizes, signal_levels=stack_signal_levels,
    levels_to_display=np.concatenate(([0,], 
                                      np.nonzero(stack_signal_levels >= 0.1)[0])),
    plot_file='DistributionVsNumberTrials_Peak.png')

  plot_distribution_analysis(distribution_dict['Covariance'], block_sizes,
                             ylabel='Covariance',
                             plot_file='DistributionAnalysis-Covariance.png')
  plot_distribution_analysis(distribution_dict['TotalRMS'], block_sizes,
                             ylabel='Total RMS',
                             plot_file='DistributionAnalysis-TotalRMS.png')
  plot_distribution_analysis(distribution_dict['TrialRMS'], block_sizes,
                             ylabel='Trial RMS',
                             plot_file='DistributionAnalysis-TrialRMS.png')
  plot_distribution_analysis(distribution_dict['Peak'], block_sizes,
                             ylabel='Peak',
                             plot_file='DistributionAnalysis-Peak.png')

  dprime_dict = calculate_all_dprime(distribution_dict, block_sizes)
  # Returns a dictionary of d' arrays, each array of size 
  #   num_trial_sizes x num_levels x num_trials

  sound_levels_to_plot = [0] + np.nonzero(stack_signal_levels >= 0.1)[0].tolist()
  sound_levels_to_plot.sort(reverse=True)  # plot biggest first for legend's order
  plot_dprime_vs_trials(dprime_dict['Covariance'], 'Covariance vs. Trial Count', 
                        sound_levels=stack_signal_levels,
                        sound_levels_to_plot=sound_levels_to_plot,
                        block_sizes=block_sizes, plot_file='DprimeVsTrialCount_Covariance.png')
  plot_dprime_vs_trials(dprime_dict['TotalRMS'], 'Total RMS vs. Trial Count', block_sizes,
                        sound_levels=stack_signal_levels,
                        sound_levels_to_plot=sound_levels_to_plot,
                        plot_file='DprimeVsTrialCount_TotalRMS.png')
  plot_dprime_vs_trials(dprime_dict['TrialRMS'], 'Trial RMS vs. Trial Count', block_sizes,
                        sound_levels=stack_signal_levels,
                        sound_levels_to_plot=sound_levels_to_plot,
                        plot_file='DprimeVsTrialCount_TrialRMS.png')
  plot_dprime_vs_trials(dprime_dict['Peak'], 'Peak vs. Trial Count', block_sizes,
                        sound_levels=stack_signal_levels,
                        sound_levels_to_plot=sound_levels_to_plot,
                        ylabel='Peak/RMS Noise Ratio',
                        plot_file='DprimeVsTrialCount_Peak.png')

  plot_dprimes_vs_sound_level_distribution(
    dprime_dict['Covariance'], block_sizes, stack_signal_levels, 
    plot_file='DPrimeVsLevel_Covariance.png')
  plot_dprimes_vs_sound_level_distribution(
    dprime_dict['Peak'], block_sizes, stack_signal_levels, 
    ylabel='Peak/RMS Noise Ratio',
    plot_file='DPrimeVsLevel_Peak.png')
  plot_dprimes_vs_sound_level_distribution(
    dprime_dict['TotalRMS'], block_sizes, stack_signal_levels, 
    plot_file='DPrimeVsLevel_TotalRMS.png')
  plot_dprimes_vs_sound_level_distribution(
    dprime_dict['TrialRMS'], block_sizes, stack_signal_levels, 
    plot_file='DPrimeVsLevel_TrialRMS.png')

  plot_dprimes_vs_sound_level(dprime_dict, stack_signal_levels)

  plot_dprimes_vs_trials(dprime_dict, block_sizes=block_sizes)

  compute_thresholds(dprime_dict['TotalRMS'], stack_signal_levels, 
                     block_sizes, 'TotalRMS')
  compute_thresholds(dprime_dict['TrialRMS'], stack_signal_levels, 
                     block_sizes, 'TrialRMS')
  compute_thresholds(dprime_dict['Covariance'], stack_signal_levels, 
                     block_sizes, 'Covariance')
  compute_thresholds(dprime_dict['Peak'], stack_signal_levels, 
                     block_sizes, 'Peak')


if __name__ == "__main__":
  app.run(main)