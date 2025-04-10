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
    self.assertAlmostEqual(dprime, 10, delta=0.1)


class MetricTests(absltest.TestCase):
  def test_rms(self):
    signal_levels = np.linspace(0, 1, 11)
    exp_stack = metrics.create_synthetic_stack(noise_level=1, 
                                               signal_levels=signal_levels)

    rms_metric = metrics.RMSMetric()
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

if __name__ == "__main__":
  absltest.main()