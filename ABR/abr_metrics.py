import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Dict, Generator, List, Optional, Tuple, Union


mouse_sample_rate = 24414 * 8  # From George's Exp Notes, 8x oversampling


# To calculate the time when the gammatone envelope reaches its peak, use 
# Wolfram Alpha to compute:
# 
# Solve[D[(t^e)*e^(-2*Pi*b*t), t] == 0, t]
#
# Result is e/(2*pi*b)

def create_synthetic_stack(noise_level=1, num_times=1952, num_trials=1026,
                           bw=200, order=4, cf=1000, signal_levels=(0, 1),
                           sample_rate=mouse_sample_rate):
    """Create a synthetic stack of ABR recordings so we can investigate d'
    behaviour for really large number of trials.  This stack has two different
    (sound pressure) levels.

    Args:
      noise_level: Amplitude of the white noise signal. 
      num_times: The length of the ABR response in samples
      num_trials: How many trials to create
      bw: The bandwidth of the envelope (in Hz)
      order: the Gammatone order
      cf: The center frequency of the carrier (in Hz)
      signal_levels: A list of signal levels to generate
      sample_rate: What sample rate to generate the signal at

    Returns:
      a 3d tensor with shape num_levels x num_times x num_trials
    """
    def gammatone_func(t, cf=cf, bw=bw, order=order):
        envelope = t ** (order - 1) * np.exp(-2 * np.pi * bw * t)
        if cf:
          return envelope * np.sin(2 * np.pi * cf * t)
        else:
          return envelope

    t = np.arange(num_times) / mouse_sample_rate
    peak_time = 3/(2*np.pi*bw)
    peak_env = gammatone_func(peak_time, cf=0)
    gammatone = gammatone_func(t)/peak_env
    # The shape of the stacks array is levels x time x trials
    stack = noise_level * np.random.normal(size=(len(signal_levels), num_times, num_trials))
    signals = np.expand_dims(signal_levels, (1, 2)) * np.expand_dims(gammatone, (0, 2))
    stack += signals
    return stack


def bootstrap_sample(data: NDArray, 
                     bootstrap_size: int) -> NDArray:
  """Grab a random subset of the trials from the data, choosing with
  replacement.
  Args:
    data: The data to sample, shape [num_levels x ] num_dims x num_total_trials
    bootstrap_size: How many samples of the data to pull from the original data.

  Returns:
    An array of size [num_levels x ] x num_dims x bookstrap_size.
    The input and output arrays have the same number of dimensions, only the
    last dimension is subsampled.
  """
  assert data.ndim >= 2
  trial_count = data.shape[-1]
  if trial_count == bootstrap_size:
     return data  # Nothing to do

  return data[..., np.random.choice(trial_count, bootstrap_size)]


def calculate_dprime(
    h1: Union[list, NDArray],
    h2: Union[list, NDArray],
    geometric_mean: bool = False,
    debug: bool = True
) -> float:
    """Calculate the d' given two sets of (one-dimensiona) data.  The h1
    data should be the bigger of the two data. The normalization factor either
    the arithmetic mean (default) of the two standard deviations, if the data is
    additive, or a geometric mean if the data is based on a multiplicative ratio.
    """
    if geometric_mean:
        return (np.mean(h1) - np.mean(h2)) / np.sqrt(np.std(h1) * np.std(h2))
    else:
        # Normalize by arithmetic mean of variances (not std)
        norm = np.sqrt((np.std(h1) ** 2 + np.std(h2) ** 2) / 2.0)
        dprime = (np.mean(h1) - np.mean(h2)) / (1e-10 + norm)
        if debug:
           print('Calculate dprime: norm', norm, 'dprime:', dprime)
           print('mean h1:', np.mean(h1), 'mean h2:', np.mean(h2))
           print('std h1:', np.std(h1), 'std h2:', np.std(h2))
        return dprime
    

class Metric(object):
  def __init__(self, *args, **kwargs):
    del args
    del kwargs

  def compute(self, stack: NDArray) -> NDArray:
    return np.array(())  # Should never be called, always specialized.

  def compute_window(
    self,
    stack: NDArray,
    window_start: int = 0,
    window_end: int = 0,
  ) -> float:
    assert stack.ndim == 2, f'Wanted two dimensions, got {stack.shape}'

    if window_start and window_end:
        stack = stack[:, window_start:window_end]
    return self.compute(stack)

  def compute_distribution(self, exp_stack: NDArray) -> NDArray:
    # Number of levels x num of time samples x number of trials
    """Look at the entire set of data, all trials.
    """
    assert exp_stack.ndim == 3, f'Wanted three dimensions, got {exp_stack.shape}'

    dist = np.zeros((exp_stack.shape[0], exp_stack.shape[2]))
    for l in range(exp_stack.shape[0]):
      dist[l, :] = self.compute(exp_stack[l, ...])
    return dist # Shape: num_levels x num_trials

  def compute_distribution_by_trial_size(self,
                                         exp_stack: NDArray,
                                         block_sizes: List[int],
                                         bootstrap_repetitions: int = 20,
                                         min_count = 10,
                                         max_count = 20000,
                                         ):
    """Compute the distribution for a stack of data as a function of trial count.
    Use bootstrapping.
    
    Returns:
      List with num_block_counts 3D arrays.  Each array of size
        num_levels x bootstrap_repetitions x trial_count
    """
    assert exp_stack.ndim == 3, f'Wanted three dimensions, got {exp_stack.shape}'
    num_levels, _, trial_count = exp_stack.shape
    bookstrap_repetitions = 20
    print('block sizes are:', block_sizes)
    block_sizes = np.asarray(block_sizes)

    block_sizes = block_sizes[(block_sizes >= min_count) & 
                              (block_sizes <= max_count)]
    dist = []

    for i, trial_count in enumerate(block_sizes):
      block_results = np.zeros((num_levels, bookstrap_repetitions, trial_count))
      for j in range(bootstrap_repetitions):
        sample = bootstrap_sample(exp_stack, trial_count)
        for l in range(exp_stack.shape[0]): # For each level...
          block_results[l, j, :] = self.compute(sample[l, ...])
      dist.append(block_results)
    return dist, block_sizes
 

class PeakMetric(Metric):
  """Look at peak amplitude of the average, which should be all signal,
  vs. the RMS energy of the noise.  We want to be some number of standard
  deviations above the noise.
  https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-10-104
  """
  def __init__(self, 
               window_start=int(1.75e-3*mouse_sample_rate), 
               window_end=int(2.75e-3*mouse_sample_rate)):
    super().__init__()
    self.window_start = window_start
    self.window_end = window_end

  def compute(self, stack: NDArray) -> NDArray:
    assert stack.ndim == 2, f'Wanted two dimensions, got {stack.shape}'
    signal_ave = np.mean(stack[self.window_start:self.window_end, :], axis=1)
    noise_ave = np.mean(shuffle_2d_array(stack), axis=1)
    snr = np.max(np.abs(signal_ave))/np.std(noise_ave)
    return np.reshape(snr, (1,))  # Need 1d array to match other metrics
  

class TotalRMSMetric(Metric):
  def compute(self, stack: NDArray) -> NDArray:
    """
    Compute the RMS of the waveform recordings, one per trial.

    Args:
      stack: 2D tensor of waveform recordings: num_times x num_trials
    
    Returns:
      A 1d distribution array of size num_trials
    """
    assert stack.ndim == 2, f'Wanted two dimensions, got {stack.shape}'
    return np.sqrt(np.mean(stack**2, axis=0))


class CovarianceMetric(Metric):
  def __init__(self, with_self_similar=False): 
    self.with_self_similar = with_self_similar

  def compute(self, 
              stack: NDArray, 
              model: Optional[NDArray] = None) -> NDArray:
    """
    Compute the matched filter output of the waveform recordings, one per trial.

    Args:
      stack: 2D tensor of waveform recordings: num_times x num_trials
      model: Optional exact form of expected signal for testing.
    
    Returns:
      A 1d distribution array of size num_trials
    """
    assert stack.ndim == 2, f'Wanted two dimensions, got {stack.shape}'
    
    if model is None:
      model = np.mean(stack, axis=-1, keepdims=True)

    if self.with_self_similar:  # Consider all the terms
      stack = np.reshape(model, (stack.shape[0], 1)) * stack
      response = np.mean(stack, axis=0)  # Sum response over time
    else:
      num_trials = stack.shape[1]
      response = np.zeros(num_trials)
      model = model[:, 0]
      for i in range(num_trials):
        model_without = (model * num_trials - stack[:, i]) / (num_trials - 1)
        response[i] = np.mean(model_without * stack[:, i])
    # Can't take sqrt since response with noise is symmetric around 0.
    # return np.sqrt(np.maximum(0, response))
    return response


class CovarianceSelfSimilarMetric(CovarianceMetric):
  def __init__(self):
     super().__init__(with_self_similar=True)
  # All the rest as is


class PrestoMetric(Metric):
  def __init__(self, num_splits=500):
    """500 splits is what is used in the paper."""
    self.num_splits = num_splits

  def compute(self, stack: NDArray) -> NDArray:
    """
    Compute a self-similarity measure based on binary splits proposed by the
    ABRpresto paper.

    Args:
      stack: 2D tensor of waveform recordings: num_times x num_trials
    """
    assert stack.ndim == 2
    correlations = np.zeros(self.num_splits)
    for i in range(self.num_splits):
      selections = np.random.uniform(size=stack.shape[1]) > 0.5
      mean1 = np.mean(stack[:, selections], axis=1)
      mean2 = np.mean(stack[:, np.logical_not(selections)], axis=1)
      correlations[i] = np.mean(mean1*mean2)/np.std(mean1)/np.std(mean2)
    return correlations


all_metrics = {
   'peak': PeakMetric,
   'total_rms': TotalRMSMetric,
   'covariance': CovarianceMetric,
   'covariance_self_similar': CovarianceSelfSimilarMetric,
   'presto': PrestoMetric
}

def shuffle_2d_array(array_2d):
    array2d = array_2d.copy()
    flat_array = array_2d.flatten()
    np.random.shuffle(flat_array)
    return flat_array.reshape(array_2d.shape)

    
def calculate_dprimes(exp_stack: NDArray, 
                      metric: Metric,
                      signal_index: int = -1,
                      noise_index: int = 0,
                      window_start: int = 0,
                      window_end: int = 0,
                      **compute_args
                      ) -> NDArray:
  assert exp_stack.ndim == 3  # num_levels x num_times x num_trials
  dprimes = np.zeros(exp_stack.shape[0])

  shuffled_noise = shuffle_2d_array(exp_stack[noise_index, ...])
  noise_dist = metric.compute_window(shuffled_noise,
                                     window_start=window_start,
                                     window_end=window_end)

  for i in range(exp_stack.shape[0]):
    signal_dist = metric.compute_window(exp_stack[i, ...], 
                                        window_start=window_start,
                                        window_end=window_end)
    dprimes[i] = calculate_dprime(signal_dist, noise_dist)
  return dprimes


def show_response_stack(
    stack: NDArray,
    levels: Union[NDArray, List[float]],
    alpha: float = 0.01,
    title: str = "",
    skip_levels: int = 3,
    relative_max: float = 1.5,
    absolute_max: float = 0,
    num_cols: int = 1,
    col_num: int = 0,
) -> None:
    """Show a stack of all ABR waveforms across levels.  The number of plots is
    determined by the number of levels in stack, and skip_levels

    Args:
      stack: a 3 dimensional tensor from create_stack for one animal of shape
        level, time, trial
      levels: which levels are defined in this stack
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
    if isinstance(levels, np.ndarray):
      levels = levels.tolist() 
    t = np.arange(stack.shape[-2]) / mouse_sample_rate
    for i, level in enumerate(levels2plot):
        plt.subplot(len(levels2plot), num_cols, i * num_cols + col_num + 1)
        plt.plot(t * 1000, stack[levels.index(level), ...], alpha=alpha)
        mean_stack = np.mean(stack[levels.index(level), ...], axis=-1)
        plt.plot(t * 1000, mean_stack, color="r")
        m = np.max(np.abs(mean_stack))
        if m == 0:
           m = np.max(np.abs(stack))  # In case waveform is all zero.
        if absolute_max > 0:
            plt.ylim(-absolute_max, absolute_max)
        else:
            plt.ylim(-relative_max * m, relative_max * m)
        wave_rms = np.sqrt(np.mean(stack[levels.index(level), ...] ** 2))
        plt.text(np.max(t * 1000) * 0.75, 1.20 * m, 
                 f"Waveform RMS={wave_rms:5.3g}")
        ave_rms = np.sqrt(np.mean(mean_stack**2))
        plt.text(0.0, 1.20 * m, f"Average RMS={ave_rms:5.3g}", color="red")

        plt.xlabel("Time (ms)")
        plt.ylabel(f"{level}dB")

        if i == 0:
            plt.title(title)
        if i != len(levels2plot) - 1:
            plt.gca().xaxis.set_tick_params(labelcolor="none")


def measure_full_stack(stack) -> Dict[str, NDArray]:
  """Calculate all metrics on a full 5-dimensional stack of waveforms.
  For each frequency, level and channels, summarize the ERP data and create a
  new 4d array adding the summary.

  Return a dictionary of 4d stack, mirroring the first 3d, and adding a 
  dimension, depending on the metric, for that metric's analysis.
  """
  assert stack.ndim == 5 # Freqs x levels x channels x time x trials
  num_freqs, num_levels, num_channels, num_time, num_trials = stack.shape
  results = {}
  for key in all_metrics:
    metric = all_metrics[key]()
    metric_result = None
    for f in range(num_freqs):
      for l in range(num_levels):
        for c in range(num_channels):
          r = metric.compute(stack[f, l, c, :, :])
          if metric_result is None:
            metric_result = np.zeros((num_freqs, num_levels, num_channels, 
                                      len(r)))
          metric_result[f,l,c] = r
    results[key] = metric_result
  return results