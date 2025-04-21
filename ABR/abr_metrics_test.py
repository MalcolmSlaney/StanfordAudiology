from collections import Counter
import math
import os

from absl.testing import absltest
import abr_metrics as metrics

import numpy as np
import matplotlib.pyplot as plt


class ABRGammatoneTests(absltest.TestCase):
  def test_gammatone(self):
    signal_levels = np.linspace(0, 1, 11)
    stack = metrics.create_synthetic_stack(num_times=4200, num_trials=52, 
                                           signal_levels=signal_levels,
                                           noise_level=0)
    self.assertEqual(stack.shape, (len(signal_levels), 4200, 52))

    measured_levels = np.max(stack, axis=(1, 2))

    np.testing.assert_allclose(signal_levels, measured_levels, rtol=0.01)

class DPrimeTests(absltest.TestCase):
  def test_dprime(self):
    signal = np.random.normal(1, .1, 10240)
    noise = np.random.normal(0, .1, 10240)

    dprime = metrics.calculate_dprime(signal, noise)
    self.assertAlmostEqual(dprime, 10, delta=0.15)

  def test_stack_dprime(self):
    exp_stack = metrics.create_synthetic_stack(noise_level=1, 
                                               signal_levels=np.linspace(0, .9, 10))
    cov_ns_metric = metrics.CovarianceMetric(with_self_similar=False)
    cov_dprimes = metrics.calculate_dprimes(exp_stack, cov_ns_metric)
    self.assertGreater(np.mean(cov_dprimes), 10)
    self.assertLess(np.std(cov_dprimes), 10)  # Why so high?

class ShuffleTests(absltest.TestCase):
  def test_shuffle(self):
    """Shuffle a big array and make sure most elements are different."""
    start_array = np.arange(100).reshape((10, 10))
    rnd_array = metrics.shuffle_2d_array(start_array)
    self.assertLess(np.sum(start_array == rnd_array), 5)


class BootstrapTests(absltest.TestCase):
  def test_bootstrap(self):
    data = np.reshape(np.arange(1000000), (2, -1))
    bootstrap_size = 10
    for _ in range(100):
      b = metrics.bootstrap_sample(data, bootstrap_size)
      self.assertEqual(b.shape, (2, bootstrap_size))
      c = Counter(np.reshape(b, (-1)))
      self.assertLess(max(c.values()), 2,
                      f'Got {c}')


class MetricTests(absltest.TestCase):
  def test_peak(self):
    num_points = 2000
    num_trials = 200

    signal = np.zeros((num_points, num_trials))
    signal[42, :] = 1.0

    noise = np.random.random((num_points,num_trials))

    response = signal + noise
    peak_metric = metrics.PeakMetric(0, -1)
    snr = peak_metric.compute(response)
    self.assertGreater(snr, 50)

    response = signal + 2*noise
    snr2 = peak_metric.compute(response)
    self.assertGreater(snr, snr2)

  def test_rms(self):
    signal_levels = np.linspace(0, 1, 11)
    exp_stack = metrics.create_synthetic_stack(noise_level=1, 
                                               signal_levels=signal_levels)

    rms_metric = metrics.TotalRMSMetric()
    rms_dist_s = rms_metric.compute(exp_stack[-1, ...])
    rms_dist_n = rms_metric.compute(exp_stack[0, ...])
    self.assertLess(np.mean(rms_dist_n), np.mean(rms_dist_s))

  def test_cov(self):
    signal_levels = np.linspace(0, 1, 11)
    exp_stack = metrics.create_synthetic_stack(noise_level=1, 
                                               signal_levels=signal_levels)

    cov_metric = metrics.CovarianceMetric()
    cov_dist_s = cov_metric.compute(exp_stack[-1, ...])
    cov_dist_n = cov_metric.compute(exp_stack[0, ...])
    self.assertLess(np.mean(cov_dist_n), np.mean(cov_dist_s))

    cov_metric = metrics.CovarianceSelfSimilarMetric()
    cov_dist_ss = cov_metric.compute(exp_stack[-1, ...])
    cov_dist_ns = cov_metric.compute(exp_stack[0, ...])
    self.assertLess(np.mean(cov_dist_ns), np.mean(cov_dist_ss))
    # Without self-similar is less than with self-similar
    self.assertLess(np.mean(cov_dist_s), np.mean(cov_dist_ss))

  def test_presto(self):
    signal_levels = np.linspace(0, 1, 11)
    exp_stack = metrics.create_synthetic_stack(noise_level=1, 
                                               signal_levels=signal_levels)

    presto_metric = metrics.PrestoMetric()
    presto_dist_s = presto_metric.compute(exp_stack[-1, :, :400])
    presto_dist_n = presto_metric.compute(exp_stack[0, :, :400])
    self.assertLess(np.mean(presto_dist_n), np.mean(presto_dist_s))

  def test_all(self):
    """Test computing all metrics on a full 5d (George@Stanford) stack."""
    signal_levels = np.linspace(0, .9, 10)
    exp_stack = metrics.create_synthetic_stack(noise_level=1, 
                                               signal_levels=signal_levels)
    self.assertEqual(exp_stack.shape, (10, 1952, 1026))
    full_stack = np.expand_dims(exp_stack, (0, 2))
    self.assertEqual(full_stack.shape, (1, 10, 1, 1952, 1026))

    all_results = metrics.measure_full_stack(full_stack)

    for k in metrics.all_metrics:
      if k == 'presto':
        self.assertEqual(all_results[k].shape, (1, 10, 1, 500))
      elif k == 'peak':
        self.assertEqual(all_results[k].shape, (1, 10, 1, 1))
      else:
        self.assertEqual(all_results[k].shape, (1, 10, 1, 1026))



class PlotTests(absltest.TestCase):
  def test_plot(self):
    plt.clf()
    signal_levels = np.linspace(0, 1, 11)
    stack = metrics.create_synthetic_stack(num_times=4200, num_trials=52, 
                                           signal_levels=signal_levels,
                                           noise_level=0)
    self.assertEqual(stack.shape, (len(signal_levels), 4200, 52))
    metrics.show_response_stack(stack, (100*signal_levels).astype(int), 
                                title='Test Plot')
    plt.savefig('test_show_response_stack.png')

if __name__ == "__main__":
  absltest.main()