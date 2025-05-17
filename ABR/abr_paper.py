import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt

from typing import Dict, Optional, Union

# Note this syntax (import *) is in general a bad idea.  But it's a good
# solution in colab so you can easily redefine a function that was already
# defined in the abr.py file.  Just don't forget to check in the new version
# at some point.
#   https://www.geeksforgeeks.org/why-import-star-in-python-is-a-bad-idea/

from ABR.abr_metrics import *
from ABR.abr_george import BilinearInterpolation, PositivePolynomial, Metric


def create_exp_stack(num_trials: int = 4096, 
                     signal_levels: List[float] = []) -> NDArray:
  if not len(signal_levels):
    signal_levels = np.linspace(0, .9, 10)
  exp_stack = create_synthetic_stack(noise_level=1,
                                     signal_levels=signal_levels,
                                     num_trials=num_trials)
  # exp_stack.shape is Number of levels, # of time samples, # of trials
  return signal_levels, exp_stack


def plot_exp_stack_waveform(exp_stack: NDArray, level_index: int = -1):
  plt.plot(exp_stack[level_index, :, 0], label='One trial')
  plt.plot(np.mean(exp_stack[-1, ...], axis=level_index, label='Average')
  plt.title('ABR Response at Highest Sound Level')
  plt.legend()


def compute_distribution(metric: Metric, exp_stack: NDArray) -> NDArray:
  assert isinstance(metric, Metric)
  # Number of levels x num of time samples x number of trials
  assert exp_stack.ndim == 3

  dist = np.zeros((exp_stack.shape[0], exp_stack.shape[2]))
  for l in exp_stack.shape[0]:
    dist[l, :] = metric.compute(exp_stack[l, ...])
  return dist


def plot_distribution_histogram_comparison(top_dist: NDArray, 
                                           bottom_dist: NDArray, 
                                           top_label: str = 'Top',
                                           bottom_label: str = 'Bottom',
                                           bin_count: int = 20
                                           ):
  assert top_dist.ndim == 2
  assert bottom_dist.ndim == 2

  plt.subplot(2, 1, 1)
  plt.hist(top_dist[0, :], bin_count, label=f'Noise ({top_label})')
  plt.hist(top_dist[-1, :], bin_count, label=f'Signal ({top_label})')
  plt.legend();

  plt.subplot(2, 1, 2)
  plt.hist(bottom_dist[0, :], bin_count, label=f'Noise ({bottom_label})')
  plt.hist(bottom_dist[-1, :], bin_count, label=f'Signal ({bottom_label})')

  plt.legend();
