import os

from absl import app, flags
import numpy as np
from numpy.typing import NDArray

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
    pickle_file = os.path.join(Synthetic_ABR_Cache_Dir, cache_file)
    with open(pickle_file, "w") as f:
        f.write(jsonpickle.encode(data))
        print(f'  Cached data for {data.keys()} data '
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
                     cache_file: str = 'exp_stack') -> NDArray:
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
  assert top_dist.ndim == 2
  assert bottom_dist.ndim == 2

  plt.clf()
  plt.subplot(2, 1, 1)
  plt.hist(top_dist[0, :], bin_count, label=f'Noise ({top_label})')
  plt.hist(top_dist[-1, :], bin_count, label=f'Signal ({top_label})')
  plt.legend();

  plt.subplot(2, 1, 2)
  plt.hist(bottom_dist[0, :], bin_count, label=f'Noise ({bottom_label})')
  plt.hist(bottom_dist[-1, :], bin_count, label=f'Signal ({bottom_label})')

  plt.legend();

  if plot_file:
     plt.savefig(plot_file)

##################  Compute a distribution and plot it  #######################

def distribution_comparison(dist_rms, dist_cov, block_sizes):
  plt.figure(figsize=(8, 6))

  plt.subplot(2, 2, 1)
  means = [np.mean(cov_dist_l9[b]) for b in range(len(block_sizes))]
  stds = [np.std(cov_dist_l9[b]) for b in range(len(block_sizes))]
  plt.errorbar(block_sizes, means, yerr=4*np.asarray(stds),
          label='Signal')
  meann = [np.mean(cov_dist_l0[b]) for b in range(len(block_sizes))]
  stdn = [np.std(cov_dist_l0[b]) for b in range(len(block_sizes))]
  plt.errorbar(block_sizes, meann, yerr=4*np.asarray(stdn),
          label='Noise')
  plt.gca().set_xscale('log')
  plt.legend()
  # plt.xlabel('Trial Count')
  plt.ylabel('Cov. of the Distribution')
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

##################   Main Program  #######################

def main(*argv):
  stack_signal_levels, exp_stack = create_exp_stack()
  plot_exp_stack_waveform(exp_stack)
   
  metric_rms = TotalRMSMetric()
  dist_rms = compute_distribution(metric_rms, exp_stack)

  metric_cov = CovarianceMetric()
  dist_cov = compute_distribution(metric_cov, exp_stack)

  plot_distribution_histogram_comparison(dist_rms, dist_cov,
                                         )

if __name__ == "__main__":
  app.run(main)