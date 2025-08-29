"""Complete simulations to support the Detecting indivual ABR paper.


Single and Multipoint.

"""
from absl import app, flags
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import List, Tuple

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
from sklearn.linear_model import Ridge, LinearRegression


def single_inner_product_model(signal: float = 1.0, 
                               noise: float = 1.0, 
                               num_trials: int = 1000, 
                               num_experiments: int = 100000,
                               self_similar: bool = True):
  """Create an ensemble of data, size num_samples,
  consisting of signal plus noise. Measure the mean and variance of the
  inner product with a known signal level (a constant).
  This is for a single point in time.
  Args:
    self_similar: if true, include the current trial in the similarity measure
      if False, exclude the current trial from the measure.
  """
  # print('Testing:', signal, noise)
  d = signal + noise*np.random.randn(num_trials, num_experiments)
  trial_signal = d[:1, :] # Pick a signal for trial
  if self_similar:
    model = np.mean(d, axis=0, keepdims=True) # Compute the model for each experiment
  else:
    model = np.mean(d[1:, :], axis=0, keepdims=True)
  assert model.shape == (1, num_experiments)
  dd = trial_signal*model # Multiply model by data, and average the result for each experiment
  return np.mean(dd), np.var(dd) # Average across all the experiments.


def single_inner_product_figure():
  trial_counts = 10**np.linspace(1, 4, num=10)
  ss_means = []
  ss_vars = []
  s=2
  n=4
  for num_trials in trial_counts:
    m, v = single_inner_product_model(signal=s, noise=n, num_trials=int(num_trials))
    print(f'{num_trials:.3f} & {m:.3f} & {v:.3f}')
    ss_means.append(m)
    ss_vars.append(v)

  plt.figure(figsize=(12, 4))
  plt.subplot(1, 3, 1)
  plt.semilogx(trial_counts, ss_means, 'x')
  single_inner_product_mean = s**2+n**2/trial_counts
  plt.semilogx(trial_counts, single_inner_product_mean)
  plt.title('Mean of Single Trial Model (Full Covariance)')

  plt.subplot(1,3,2)
  plt.semilogx(trial_counts, ss_vars, 'x')
  # single_inner_product_variance = (s**2)*(n**2)*(1+2/trial_counts+1/(trial_counts**2)) + (trial_counts + 1)*(n**4)/(trial_counts**2)
  single_inner_product_variance = (s**2)*(n**2)*(1+3/trial_counts) + (trial_counts + 1)*(n**4)/(trial_counts**2)
  plt.semilogx(trial_counts, single_inner_product_variance)
  plt.title('Variance of Single Trial Model (Full Covariance)')

  plt.subplot(1,3,3)
  inner_product_dprime = single_inner_product_mean / (2*np.sqrt(single_inner_product_variance))
  plt.semilogx(trial_counts, np.asarray(ss_means)/(2*np.sqrt(np.asarray(ss_vars))), 'x');
  plt.semilogx(trial_counts, inner_product_dprime)
  plt.title('d\' of Single TrialModel (Full Covariance)');

  plt.savefig('single_trial_inner_product_comparison_full_covariance.png')


def single_inner_product_figure_jackknife():
  trial_counts = 10**np.linspace(1, 4, num=10)
  noss_means = []
  noss_vars = []
  s=2
  n=4
  for num_trials in trial_counts:
    m, v = single_inner_product_model(signal=s, noise=n, 
                                      num_trials=int(num_trials),
                                      self_similar=False)
    print(f'{num_trials:.3f} & {m:.3f} & {v:.3f}')
    noss_means.append(m)
    noss_vars.append(v)
    plt.figure(figsize=(12, 4))
  plt.subplot(1, 3, 1)
  plt.semilogx(trial_counts, noss_means, 'x')
  single_inner_product_mean = s**2 + 0*trial_counts
  plt.semilogx(trial_counts, single_inner_product_mean)
  plt.title('Mean of Single Trial Model')

  plt.subplot(1,3,2)
  plt.semilogx(trial_counts, noss_vars, 'x')
  single_inner_product_variance = (s**2)*(n**2)*(1+1/(trial_counts-1)) + n**4/(trial_counts-1)
  plt.semilogx(trial_counts, single_inner_product_variance)
  plt.title('Variance of Single Trial Model')

  plt.subplot(1,3,3)
  inner_product_dprime = single_inner_product_mean / (2*np.sqrt(single_inner_product_variance))
  plt.semilogx(trial_counts, np.asarray(noss_means)/(2*np.sqrt(np.asarray(noss_vars))), 'x');
  plt.semilogx(trial_counts, inner_product_dprime)
  plt.title('d\' of Single TrialModel');

  plt.savefig('single_trial_inner_product_comparison_jackknife.png')


def multi_inner_product_model(signal: ArrayLike = [1.0,], 
                               noise: float = 1.0, 
                               num_trials: int = 1000, 
                               num_experiments: int = 100000,
                              ):
  """Create an ensemble of data, size num_samples,
  consisting of signal plus noise. Measure the mean and variance of the
  inner product with a known signal level (a constant).
  This is for a single point in time.
  Args:
    self_similar: if true, include the current trial in the similarity measure
      if False, exclude the current trial from the measure.
  """
  # print('Testing:', signal, noise)
  num_signal_points = len(signal)
  signal = np.asarray(signal).reshape((num_signal_points, 1, 1))
  d = signal + noise*np.random.randn(num_signal_points, num_trials, num_experiments)
  dd = np.sum(d*signal, axis=0) # Dot product over signal
  # We want the covariance of each trial's dot product, so don't average!
  covariance = dd # np.mean(dd, axis=0)  # Average over the trials
  # assert covariance.shape == (num_experiments,)
  return np.mean(covariance), np.var(covariance), d  # Stats over experiemnts


def multi_inner_product_figure():
  num_trials = 2000
  wav_means = []
  wav_vars = []
  s= np.asarray([2]) # 
  s= np.asarray([1,2,1,0,-1,-2,-1])
  n_s = np.asarray([1,2,3,4])
  for n in n_s:
    m, v, d = multi_inner_product_model(signal=s, noise=n, 
                                      num_experiments=100,
                                      num_trials=num_trials,
                                    )
    print(f'{num_trials:.3f} & {m:.3f} & {v:.3f}')
    wav_means.append(m)
    wav_vars.append(v)
    
    plt.figure(figsize=(12, 4))
  plt.subplot(1, 3, 1)
  plt.plot(n_s, wav_means, 'x', label='Simulation')
  plt.plot(n_s, np.zeros_like(n_s) + np.sum(s**2), label='Prediction')
  plt.ylim(np.sum(s**2)*.99, np.sum(s**2)*1.01)
  plt.xlabel('Noise Level ($\\sigma$)')
  plt.title('Mean of Waveform Model')

  plt.subplot(1,3,2)
  plt.plot(n_s, wav_vars, 'x', label='Simulated')
  plt.plot(n_s, n_s**2*np.sum(s**2), label='Prediction')
  plt.xlabel('Noise Level ($\\sigma$)')
  plt.title('Variance of Waveform Model')

  plt.subplot(1,3,3)
  # inner_product_dprime = wav_means / (2*np.sqrt(wav_vars))
  plt.plot(n_s, np.asarray(wav_means)/(2*np.sqrt(np.asarray(wav_vars))), 'x');
  plt.plot(n_s, np.sqrt(np.sum(s**2))/2/n_s)
  plt.xlabel('Noise Level ($\\sigma$)')
  plt.title('d\' of Waveform Model');

  plt.savefig('waveform_inner_product_comparison.png');

def main(_):
  single_inner_product_figure()
  single_inner_product_figure_jackknife()
  multi_inner_product_figure()


if __name__ == "__main__":
  app.run(main)
