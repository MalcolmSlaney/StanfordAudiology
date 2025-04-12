import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union


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


def bootstrap(data: np.ndarray, 
              bootstrap_size: int, 
              num_samples: int) -> np.ndarray:
  """Grab a random subset of the trials from the data, choosing with
  replacement.
  Args:
    data: The data to pull from, shape num_dims x num_total_trials
    bootstrap_size: How many samples of the data to pull from the original data.
    num_samples: How many bootstrap points to pull before stopping

  Returns:
    A total of num_samples arrays of size num_dims x bookstrap_size
  """
  assert data.ndim == 2
  trial_count = data.shape[1]

  for _ in range(num_samples):
    yield data[:, np.random.choice(trial_count, bootstrap_size)]


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
    signal_index: int,
    noise_index: int = 0,
    window_start: int = 0,
    window_end: int = 0,
  ) -> float:
    assert stack.ndim == 2  # num_times x num_trials

    if window_start and window_end:
        stack = stack[:, window_start:window_end]
    return self.compute(stack)


class RMSMetric(Metric):
  def compute(self, stack: np.ndarray) -> np.ndarray:
    """
    Compute the RMS of the waveform recordings, one per trial.

    Args:
      stack: 2D tensor of waveform recordings: num_times x num_trials
    """
    assert stack.ndim == 2
    return np.sqrt(np.mean(stack**2, axis=0))


class CovMetric(Metric):
  def compute(self, stack: np.ndarray) -> np.ndarray:
    """
    Compute the matched filter output of the waveform recordings, one per trial.

    Args:
      stack: 2D tensor of waveform recordings: num_times x num_trials
    """
    assert stack.ndim == 2
    signal_model = np.mean(stack, axis=-1, keepdims=True)
    return np.sqrt(np.maximum(0, np.mean(stack*signal_model, axis=0)))


class PrestoMetric(Metric):
  num_splits = 500 # From the paper
  def compute(self, stack: np.ndarray) -> np.ndarray:
    """
    Compute a self-similarity measure based on binary splits proposed by the
    ABRpresto paper.

    Args:
      stack: 2D tensor of waveform recordings: num_times x num_trials
    """
    assert stack.ndim == 2
    correlations = np.zeros(PrestoMetric.num_splits)
    for i in range(PrestoMetric.num_splits):
      selections = np.random.uniform(size=stack.shape[1]) > 0.5
      mean1 = np.mean(stack[:, selections], axis=1)
      mean2 = np.mean(stack[:, np.logical_not(selections)], axis=1)
      correlations[i] = np.mean(mean1*mean2)/np.std(mean1)/np.std(mean2)
    return correlations
