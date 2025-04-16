import numpy as np
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


def bootstrap_sample(data: np.ndarray, 
                     bootstrap_size: int) -> np.ndarray:
  """Grab a random subset of the trials from the data, choosing with
  replacement.
  Args:
    data: The data to pull from, shape num_dims x num_total_trials
    bootstrap_size: How many samples of the data to pull from the original data.

  Returns:
    An array of size num_dims x bookstrap_size
  """
  assert data.ndim == 2
  trial_count = data.shape[1]

  return data[:, np.random.choice(trial_count, bootstrap_size)]


def calculate_dprime(
    h1: Union[list, np.ndarray],
    h2: Union[list, np.ndarray],
    geometric_mean: bool = False,
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
        return (np.mean(h1) - np.mean(h2)) / norm
    

class Metric(object):
  def __init__(self, *args, **kwargs):
    del args
    del kwargs

  def compute(self, stack: np.ndarray) -> np.ndarray:
    return np.array(())  # Should never be called, always specialized.

  def compute_window(
    self,
    stack: np.ndarray,
    window_start: int = 0,
    window_end: int = 0,
  ) -> float:
    assert stack.ndim == 2  # num_times x num_trials

    if window_start and window_end:
        stack = stack[:, window_start:window_end]
    return self.compute(stack)


class TotalRMSMetric(Metric):
  def compute(self, stack: np.ndarray) -> np.ndarray:
    """
    Compute the RMS of the waveform recordings, one per trial.

    Args:
      stack: 2D tensor of waveform recordings: num_times x num_trials
    """
    assert stack.ndim == 2
    return np.sqrt(np.mean(stack**2, axis=0))


class CovarianceMetric(Metric):
  def __init__(self, with_self_similar=False): 
    self.with_self_similar = with_self_similar

  def compute(self, 
              stack: np.ndarray, 
              model: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the matched filter output of the waveform recordings, one per trial.

    Args:
      stack: 2D tensor of waveform recordings: num_times x num_trials
      model: Optional exact form of expected signal for testing.
    """
    assert stack.ndim == 2
    
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
    return np.sqrt(np.maximum(0, response))


class CovarianceSelfSimilarMetric(CovarianceMetric):
  def __init__(self):
     super().__init__(with_self_similar=True)
  # All the rest as is


class PrestoMetric(Metric):
  def __init__(self, num_splits=500):
    """500 splits is what is used in the paper."""
    self.num_splits = num_splits

  def compute(self, stack: np.ndarray) -> np.ndarray:
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

    
def calculate_dprimes(exp_stack: np.ndarray, 
                      metric: Metric,
                      signal_index: int = -1,
                      noise_index: int = 0,
                      window_start: int = 0,
                      window_end: int = 0,
                      **compute_args
                      ) -> np.ndarray:
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
    stack: np.ndarray,
    levels: Union[np.ndarray, List[float]],
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


def measure_full_stack(stack) -> Dict[str, np.ndarray]:
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