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
    cov_ns_metric = metrics.CovMetric(with_self_similar=False)
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

    cov_metric = metrics.CovMetric()
    cov_dist_s = cov_metric.compute(exp_stack[-1, ...])
    cov_dist_n = cov_metric.compute(exp_stack[0, ...])
    self.assertLess(np.mean(cov_dist_n), np.mean(cov_dist_s))

  def test_presto(self):
    signal_levels = np.linspace(0, 1, 11)
    exp_stack = metrics.create_synthetic_stack(noise_level=1, 
                                               signal_levels=signal_levels)

    presto_metric = metrics.PrestoMetric()
    presto_dist_s = presto_metric.compute(exp_stack[-1, :, :400])
    presto_dist_n = presto_metric.compute(exp_stack[0, :, :400])
    self.assertLess(np.mean(presto_dist_n), np.mean(presto_dist_s))


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