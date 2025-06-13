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
    self.assertGreater(np.mean(cov_dprimes), 9)
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
  def XXtest_peak(self):
    num_points = 2000
    num_trials = 200

    signal = np.zeros((num_points, num_trials))
    signal[42, :] = 1.0

    noise = np.random.random((num_points,num_trials))

    response = signal + noise
    peak_metric = metrics.PeakMetric(0, -1)
    snr = peak_metric.compute(response)
    self.assertGreater(snr, 5)

    response = signal + 2*noise
    snr2 = peak_metric.compute(response)
    self.assertGreater(snr, snr2)

  def XXtest_rms(self):
    signal_levels = np.linspace(0, 1, 11)
    exp_stack = metrics.create_synthetic_stack(noise_level=1, 
                                               signal_levels=signal_levels)

    rms_metric = metrics.TotalRMSMetric()
    rms_dist_s = rms_metric.compute(exp_stack[-1, ...])
    rms_dist_n = rms_metric.compute(exp_stack[0, ...])
    self.assertLess(np.mean(rms_dist_n), np.mean(rms_dist_s))

  def XXtest_cov(self):
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

  def XXtest_presto(self):
    signal_levels = np.linspace(0, 1, 11)
    exp_stack = metrics.create_synthetic_stack(noise_level=1, 
                                               signal_levels=signal_levels)

    presto_metric = metrics.PrestoMetric()
    presto_dist_s = presto_metric.compute(exp_stack[-1, :, :400])
    presto_dist_n = presto_metric.compute(exp_stack[0, :, :400])
    self.assertLess(np.mean(presto_dist_n), np.mean(presto_dist_s))

  def XXtest_all_sizes(self):
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
        self.assertEqual(
          all_results[k].shape, (1, 10, 1, 500),
          f'Size failed for presto metric. Got {all_results[k].shape}')
      elif k == 'peak':
        self.assertEqual(
          all_results[k].shape, (1, 10, 1, 1),
          f'Size failed for peak metric. Got {all_results[k].shape}')
      elif k == 'total_rms':
        self.assertEqual(
          all_results[k].shape, (1, 10, 1, 1),
          f'Size failed for RMS metric. Got {all_results[k].shape}')
      else: 
        self.assertEqual(
           all_results[k].shape, (1, 10, 1, 1026),
           f'Size failed for {k} metric. Got {all_results[k].shape}')

  def test_metric_peak(self):
    print('\nTesting Metric Peak...')
    t = np.arange(8000)/8000
    s = np.expand_dims(np.sin(2*np.pi*100*t), 1)
    # shape is num_samples x num_trials(1)
    m = metrics.PeakMetric(window_start=0, window_end=len(s))
    r = m.compute_difference(s, s)
    self.assertAlmostEqual(r, 1/(np.sqrt(2)/2))
    r = m.compute_difference(2*s, 2*s)
    self.assertAlmostEqual(r, 1/(np.sqrt(2)/2))  # Doesn't change with level
    r = m.compute_difference(2*s, s)
    self.assertAlmostEqual(r, 2/(np.sqrt(2)/2))  # Doubles if signal doubles

    exp_data = np.concatenate((np.expand_dims(s, 0),
                               np.expand_dims(2*s, 0)),
                              axis=0)
    # Shape is num_levels(2) x num_samples x num_trials(1)
    r, block_sizes = m.compute_distance_by_trial_size(exp_data, [2], 3, min_count=1)
    print('peak_metric compute_distance returned:', r)

  def test_trial_rms(self):
    t = np.arange(100)/100
    s = np.expand_dims(np.sin(2*np.pi*10*t), 1)
    noise_data = np.concatenate((0*s, 0*s, 0*s, 0*s, 0*s, 0*s, 0*s, 0*s), axis=1)
    noise_data = noise_data + np.random.randn(*noise_data.shape)
    signal_data = np.concatenate((4*s, 4*s, 4*s, 4*s, 4*s, 4*s, 4*s, 4*s), axis=1)
    signal_data = signal_data + np.random.randn(*signal_data.shape)
    m = metrics.TrialRMSMetric()
    r = m.compute_difference(signal_data, noise_data)
    print('TrialRMS.compute_difference:', r)
    self.assertGreater(r, 15) # Empirically determined
    r = m.compute_difference(2*signal_data, noise_data)
    print('TrialRMS.compute_difference x 2:', r)
    self.assertGreater(r, 25) # Empirically determined

    exp_data = np.concatenate((np.expand_dims(noise_data, 0),
                               2*np.expand_dims(signal_data, 0)),
                              axis=0)
    r = m.compute_distance_by_trial_size(exp_data, [2, ], 3, min_count=1)
    print('trial_rms compute_distance returned:', r)

  def test_covariance(self):
    t = np.arange(8000)/8000
    s = np.expand_dims(np.sin(2*np.pi*10*t), 1)
    noise_data = np.concatenate((0*s, 0*s, 0*s, 0*s, 0*s, 0*s, 0*s, 0*s), axis=1)
    noise_data = noise_data + 10*np.random.randn(*noise_data.shape)
    signal_data = np.concatenate((4*s, 4*s, 4*s, 4*s, 4*s, 4*s, 4*s, 4*s), axis=1)
    signal_data = signal_data + 10*np.random.randn(*signal_data.shape)
    m = metrics.CovarianceMetric()
    r1 = m.compute_difference(signal_data, noise_data)
    print('Covariance.compute_difference:', r1)
    self.assertGreater(r1, 10) # Empirically determined
    r2 = m.compute_difference(2*signal_data, noise_data)
    print('Covariance.compute_difference x 2:', r2)
    self.assertGreater(r2, 20) # Empirically determined

    exp_data = np.concatenate((np.expand_dims(noise_data, 0),
                               2*np.expand_dims(signal_data, 0)),
                              axis=0)
    r = m.compute_distance_by_trial_size(exp_data, [2], 3, min_count=1)
    print('trial_rms compute_distance returned:', r)


  def test_total_rms(self):
    t = np.arange(8000)/8000
    s = np.expand_dims(np.sin(2*np.pi*100*t), 1)
    noise_data = np.concatenate((1*s, 1*s, 1*s), axis=1)
    signal_data = np.concatenate((2*s, 2*s, 2*s), axis=1)
    m = metrics.TotalRMSMetric()
    r = m.compute_difference(signal_data, noise_data)
    self.assertAlmostEqual(r, 2.0)
    r = m.compute_difference(2*signal_data, noise_data)
    self.assertAlmostEqual(r, 4.0)

    exp_data = np.concatenate((np.expand_dims(noise_data, 0),
                               2*np.expand_dims(signal_data, 0)),
                              axis=0)
    r = m.compute_distance_by_trial_size(exp_data, [2], 3, min_count=1)
    print('total_rms compute_distance returned:', r)

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