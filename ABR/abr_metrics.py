from typing import Tuple

import numpy as np


class Metric(object):
  def __init__(self, *args, **kwargs):
    del args
    del kwargs

  def compute_metric(
    self, stack: np.ndarray, signal_index: int, noise_index: int = 0
  ) -> float:
    return 0.0  # Should never be called, always specialized.

  def compute(
    self,
    stack: np.ndarray,
    signal_index: int,
    noise_index: int = 0,
    window_start: int = 0,
    window_end: int = 0,
  ) -> float:
    assert stack.ndims == 3  # num_times x num_trials

    if window_start and window_end:
        stack = stack[:, window_start:window_end]
    return self.compute_metric(stack, signal_index, noise_index)

  def bootstrap_compute(
    self,
    stack: np.ndarray,
    trial_count: int,
    signal_index: int,
    repetition_count: int = 20,
    noise_index: int = 0,
  ) -> Tuple[float, float]:
    dps = []
    for j in range(repetition_count):
      # Note: transpose the resulting array slices because of this answer:
      #  https://stackoverflow.com/a/71489304
      bs_stack = stack[:, :, np.random.choice(trial_count, trial_count)].T

      dps.append(self.compute(bs_stack, signal_index, noise_index))
    return float(np.mean(dps)), float(np.std(dps))


class SNRMetric(Metric):
  def compute_metric(
      self, stack: np.ndarray, signal_index: int, noise_index: int = 0
    ) -> np.ndarray:
    """
    Args:
      stack: 3D tensor of waveform recordings: 
        num_levels x num_times x num_trials
    """
    assert stack.ndims == 3
    signal_ave = np.mean(stack[signal_index, :, :], axis=-1)
    noise_ave = np.mean(stack[noise_index, :, :], axis=-1)
    signal_rms = np.sqrt(np.mean(signal_ave**2))
    noise_rms = np.sqrt(np.mean(noise_ave**2))
    return float(signal_rms / noise_rms)


# class SNR_dprime(Metric):
