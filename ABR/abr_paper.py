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

######################  Cache Partial Results #############################

from abr_metrics import *
from abr_george import BilinearInterpolation, PositivePolynomial

# We save the raw data with Pickle because raw JSON does not support Numpy
#  https://jsonpickle.github.io/
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()


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
                     cache_file: str = 'exp_stack.pkl') -> NDArray:
  """Compute a stack of simulated ABR data using a Gammatone model.
   
  Returns: 
    A NDArray of size num_levels x num_times x num_trials
  """
  if cache_exists(cache_file):
    data = restore_from_cache(cache_file)
    return data['signal_levels'], data['waveforms']

  if not len(signal_levels):
    signal_levels = np.linspace(0, .9, 10)
  exp_stack = create_synthetic_stack(noise_level=1,
                                     signal_levels=signal_levels,
                                     num_times=num_times,
                                     num_trials=num_trials)
  # exp_stack.shape is Number of levels, # of time samples, # of trials
  save_to_cache({'signal_levels': signal_levels,
                 'waveforms': exp_stack}, cache_file)
  return signal_levels, exp_stack


def plot_exp_stack_waveform(exp_stack: NDArray, level_index: int = -1,
                            plot_file: Optional[str] ='waveform_display.png'):
  plt.clf()
  plt.plot(exp_stack[level_index, :, 0], label='One trial')
  plt.plot(np.mean(exp_stack[-1, ...], axis=level_index), label='Average')
  plt.title('ABR Response at Highest Sound Level')
  plt.legend()
  if plot_file:
    plt.savefig(plot_file)

##################  Compute a distribution and plot it  #######################


def plot_distribution_histogram_comparison(top_dist: NDArray, 
                                           bottom_dist: NDArray, 
                                           top_label: str = 'Top',
                                           bottom_label: str = 'Bottom',
                                           bin_count: int = 20,
                                           plot_file: Optional[str] = 'histogram_comparison.png',
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
     plt.savefig(plot_file)

##################  Compute a distribution and plot it  #######################

distribution_names = {
   'RMS': TotalRMSMetric,
   'Covariance': CovarianceMetric,
   'Peak': PeakMetric,
   # 'Presto': PrestoMetric # always has 500 splits!
}

def compute_all_distributions(exp_stack: NDArray, block_sizes):
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
       block_sizes) = metric.compute_distribution_by_trial_size(exp_stack,
                                                                block_sizes)
      # num_levels x bootstrap_repetitions x trial_count
      assert distribution.ndim == 3, f'Expected three dimensions in distribution, got {distribution.shape}'
      distributions[distribution_name] = distribution
      save_to_cache({'distribution': distribution,
                     'block_sizes': block_sizes}, 
                    cache_file)
  return distributions, block_sizes

def plot_distribution_vs_trials(
    distribution_list: List[NDArray], 
    block_sizes: List[int] = [], 
    plot_file: str = 'Distribution_vs_number_of_trials.png'):
  """Just for the peak metric..."""
  # num_levels x bootstrap_repetitions x trial_count
  assert isinstance(distribution_list, list), f'Expected a list of arrays, got {type(distribution_list)}'
  assert distribution_list[0].ndim == 3, f'Expected three dimensions in distribution, got {distribution_list.shape}'

  means = [np.expand_dims(np.mean(d, axis=(1,2)), 1) for d in distribution_list]
  stds = [np.expand_dims(np.std(d, axis=(1,2)), 1)  for d in distribution_list]
  means = np.concatenate(means, axis=1)
  stds = np.concatenate(stds, axis=1)
  plt.clf()
  print('Sizes', means.shape, stds.shape, len(block_sizes))
  for i in range(means.shape[0]):
    plt.errorbar(block_sizes, means[i, :], yerr=stds[i, :], label=i)
  plt.xlabel('Number of Trials')
  plt.ylabel('???')
  plt.gca().set_yscale('log')
  plt.gca().set_xscale('log')
  plt.axhline(3, ls=':')
  plt.title(f'Distribution vs. Number of Trials ')
  plt.legend()
  plt.savefig(plot_file)


def plot_distribution_analysis(dist_list, block_sizes, 
                                      plot_file: str = 'DistributionAnalysis.png'):
  # distribution is List with num_block_counts 3D arrays.  Each array of size
  #       num_levels x bootstrap_repetitions x trial_count
  plt.figure(figsize=(8, 6))

  plt.subplot(2, 2, 1)
  means = [np.mean(dist_list[b][-1, ...]) for b in range(len(block_sizes))]
  stds = [np.std(dist_list[b][-1, ...]) for b in range(len(block_sizes))]
  plt.errorbar(block_sizes, means, yerr=4*np.asarray(stds),
          label='Signal')
  meann = [np.mean(dist_list[b][0, ...]) for b in range(len(block_sizes))]
  stdn = [np.std(dist_list[b][0, ...]) for b in range(len(block_sizes))]
  plt.errorbar(block_sizes, meann, yerr=4*np.asarray(stdn),
          label='Noise')
  plt.gca().set_xscale('log')
  plt.legend()
  # plt.xlabel('Trial Count')
  plt.ylabel('Distribution of the Covariance')
  plt.title(r'Cov. Distributions - Level 9 (errorbars are $4\sigma$)');

  plt.subplot(2, 2, 2)
  dprime = ((np.asarray(means) - np.asarray(meann)) /
            (np.sqrt((np.asarray(stds)**2 + np.asarray(stdn)**2)/2)))
  plt.plot(block_sizes, dprime)
  plt.gca().set_xscale('log')
  plt.xlabel('Trial Count')
  plt.ylabel('d\'');
  # Conclusion: d' grows because noise covariance distributions gets closer to 0.

  plt.subplot(2, 2, 3)
  plt.plot(block_sizes, means, label='Signal')
  plt.plot(block_sizes, meann, label='Noise')
  plt.gca().set_xscale('log')
  plt.legend()
  plt.title('Means of Cov. Distributions')
  # plt.xlabel('Trial Count');

  plt.subplot(2, 2, 4)
  plt.plot(block_sizes, stds, label='Signal')
  plt.plot(block_sizes, stdn, label='Noise')
  plt.gca().set_xscale('log')
  plt.legend()
  plt.title(r'$\sigma$ of Cov. Distribution')
  plt.xlabel('Trial Count');

  plt.subplots_adjust(wspace=0.3, hspace=0.4);

  plt.savefig(plot_file)

##################   D' Analysis  #######################

def calculate_all_dprime(
    distribution_dict: Dict[str, List[NDArray]],
    cache_file: str = 'all_dprimes.pkl') -> Dict[str, NDArray]:
  """
  Returns
    Dictionary of d' arrays, each array of size 
      num_block_sizes x num_signal_levels
  """
  if cache_exists(cache_file):
    data = restore_from_cache(cache_file)
    return data['dprime_dict']

  dprime_dict = {}
  for distribution_name in distribution_dict:
    distribution_list = distribution_dict[distribution_name]
    dprimes = None
    block_sizes = []  # Recompute from the data
    for i in range(len(distribution_list)):  # Over block sizes
      distribution = distribution_list[i]
      block_sizes.append(distribution.shape[-1])
      assert distribution.ndim == 3, f'Expected three dimensions in distribution, got {distribution.shape}'
      num_levels, num_bootstraps, num_trials = distribution.shape
      if dprimes is None:
        # Want num_trial_sizes x num_levels x num_trials
        dprimes = np.zeros((len(distribution_list), num_levels, num_bootstraps))
      for j in range(1, num_levels):
        for k in range(num_bootstraps):
          dprimes[i, j, k] = calculate_dprime(distribution[j, k, :], 
                                              distribution[0, k, :])
    dprime_dict[distribution_name] = dprimes
    # print(f'Block size {i}: ', dprimes)
    # print(distribution_name, dprimes)
    dprimes = None

  print('block sizes are', block_sizes)
  save_to_cache({'dprime_dict': dprime_dict,
                 'block_sizes': block_sizes}, cache_file)
  return dprime_dict

def plot_dprime_result(dprimes, name='', block_sizes: List[int] = [],
                       plot_file: str = 'dprime.png'):
  # Expect num_trial_sizes x num_levels x num_trials
  plt.clf()
  dprimes = np.mean(dprimes, axis=2)  # Now num_trial_sizes x num_levels
  if len(block_sizes):
    plt.semilogx(block_sizes, dprimes)
  else:
    plt.plot(dprimes)
  plt.xlabel('Number of Trials')
  plt.legend([f'Level={l}' for l in range(dprimes.shape[1])])
  plt.title(name)
  plt.savefig(plot_file)

def plot_dprimes_vs_sound_level(
    dprime_dict: Dict[str, NDArray], 
    signal_levels: ArrayLike,
    plot_file: str = 'Covariance_RMS_Distribution_Separation.png') -> None:
  cov_dprimes = dprime_dict['Covariance']
  rms_dprimes = dprime_dict['RMS']
  plt.clf()
  # XXX_dprimes are sized: num_trial_sizes x num_levels x num_trials
  print('cov_dprimes are', cov_dprimes[0, :, :])
  print('cov_dprimes mean', np.nanmean(cov_dprimes[0, :, :], axis=-1))
  plt.plot(signal_levels, np.nanmean(cov_dprimes[0, :, :], axis=-1), label='Covariance')
  plt.plot(signal_levels, np.nanmean(rms_dprimes[0, :, :], axis=-1), label='RMS')
  plt.xlabel('Sound Level (linear a.u.)')
  plt.ylabel('d\'')
  plt.title('Distribution Separation vs. Sound Level')
  plt.legend();
  plt.savefig(plot_file)

def plot_dprimes_vs_trials(
    dprime_dict: Dict[str, NDArray], 
    level_num: int = -1, block_sizes: List[int] = [], 
    plot_file: str = 'dprimes_vs_number_of_trials.png'):
  # Expect num_trial_sizes x num_levels x num_trials
  cov_dprimes = dprime_dict['Covariance']
  assert cov_dprimes.ndim == 3, f'Expected three dimensions in cov_dprimes, got {cov_dprimes.shape}'
  rms_dprimes = dprime_dict['RMS']
  assert rms_dprimes.ndim == 3, f'Expected three dimensions in rms_dprimes, got {rms_dprimes.shape}'
  plt.clf()
  plt.semilogx(block_sizes, np.mean(cov_dprimes[:, level_num, :], axis=-1), label='Coveriance')
  plt.semilogx(block_sizes, np.mean(rms_dprimes[:, level_num, :], axis=-1), label='RMS')
  plt.xlabel('Number of Trials')
  plt.ylabel('d\'')
  plt.title(f'd\' Separation vs. Number of Trials (level {level_num})')
  plt.legend()
  plt.savefig(plot_file)

##################   Main Program  #######################

def main(*argv):
  stack_signal_levels, exp_stack = create_exp_stack()
  num_levels, num_times, num_trials = exp_stack.shape
  plot_exp_stack_waveform(exp_stack)
   
  metric_rms = TotalRMSMetric()
  dist_rms = metric_rms.compute_distribution(exp_stack)

  metric_cov = CovarianceMetric()
  dist_cov = metric_cov.compute_distribution(exp_stack)

  metric_peak = PeakMetric()
  dist_peak = metric_peak.compute_distribution(exp_stack)


  num_divisions = 14
  block_sizes = (num_trials / (2 ** np.arange(0, 
                                              num_divisions, 
                                              1.0))).astype(int)

  distribution_dict, block_sizes = compute_all_distributions(exp_stack, block_sizes)
  plot_distribution_histogram_comparison(
    distribution_dict['Covariance'][-1], distribution_dict['Covariance'][0], 
    top_label=f'trial count={block_sizes[-1]}', 
    bottom_label=f'trial count={block_sizes[0]}')

  plot_distribution_vs_trials(
    distribution_dict['Peak'], block_sizes, 
    plot_file='peak_distribution_vs_number_of_trials.png')

  plot_distribution_analysis(distribution_dict['Covariance'], block_sizes)

  dprime_dict = calculate_all_dprime(distribution_dict)
  plot_dprime_result(dprime_dict['Covariance'], 'Covariance d\'', block_sizes,
                     plot_file='Dprime_Covariance.png')
  plot_dprime_result(dprime_dict['RMS'], 'RMS d\'', block_sizes,
                     plot_file='Dprime_RMS.png')

  plot_dprimes_vs_sound_level(dprime_dict, stack_signal_levels)

  plot_dprimes_vs_trials(dprime_dict, block_sizes=block_sizes)

if __name__ == "__main__":
  app.run(main)