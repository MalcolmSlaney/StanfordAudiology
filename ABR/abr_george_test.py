import math
import os
import shutil

from absl.testing import absltest
import abr_george as george
import abr_metrics as metrics

import numpy as np
import matplotlib.pyplot as plt

test_csv_content_1 = """group,sgi,channel,subject,ref1,ref2,memo,Freq(Hz),Level(dB)	
0,52,1,control1_pre4,,,,16000,30	
0,1,2,3,4,5,6,7,8,9
0,1,2,3,4,5,6,7,8,9
0,1,2,3,4,5,6,7,8,9
0,1,2,3,4,5,6,7,8,9
"""

test_csv_content_2 = """group,sgi,channel,subject,ref1,ref2,memo,Freq(Hz),Level(dB)	
0,52,1,control1_pre4,,,,16000,10	
0,0,0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0,0,0
"""

class ABRGeorgeTests(absltest.TestCase):
  csv_dir = '/tmp'
  csv_name_1 = os.path.join(csv_dir, '20230824_control1_pre4-0-52-1-1.csv')
  csv_name_2 = os.path.join(csv_dir, '20230824_control1_pre4-0-52-1-2.csv')
  csv_name_3 = os.path.join(csv_dir, '20230824_control2_pre4-0-52-1-2.csv')

  cache_dir = '/tmp/cache'
  shutil.rmtree(cache_dir, ignore_errors=True)
  os.makedirs(cache_dir)

  def setUp(self):
    with open(ABRGeorgeTests.csv_name_1, 'w') as fp:
      fp.write(test_csv_content_1)
    with open(ABRGeorgeTests.csv_name_2, 'w') as fp:
      fp.write(test_csv_content_2)

  def test_read_exp(self):
    exp = george.read_mouse_exp(ABRGeorgeTests.csv_name_1)
    self.assertEqual(exp.filename, ABRGeorgeTests.csv_name_1)
    self.assertEqual(exp.basename, os.path.basename(ABRGeorgeTests.csv_name_1))
    self.assertEqual(exp.sgi, 52)
    self.assertEqual(exp.channel, 1)
    self.assertEqual(exp.freq, 16000)
    self.assertEqual(exp.level, 30)

    self.assertEqual(exp.single_trials.shape, (10, 4))  # time x trial

    self.assertEqual(george.exp_type_from_name(exp.basename), 'control1_pre4')

  def test_read_csv_dir(self):
    george.cache_all_mouse_dir(ABRGeorgeTests.csv_dir, ABRGeorgeTests.cache_dir)
    all_exps = george.load_waveform_cache(ABRGeorgeTests.cache_dir)
    self.assertLen(all_exps, 2)
    for exp in all_exps:
      if exp.filename == ABRGeorgeTests.csv_name_2:
        break
    self.assertIsInstance(exp, george.MouseExp)

    self.assertEqual(exp.filename, ABRGeorgeTests.csv_name_2)
    self.assertEqual(exp.basename, os.path.basename(ABRGeorgeTests.csv_name_2))
    self.assertEqual(exp.sgi, 52)
    self.assertEqual(exp.channel, 1)
    self.assertEqual(exp.freq, 16000)
    self.assertEqual(exp.level, 10)  # Different from above.

    exps = george.find_exp(all_exps, freq=16000)
    self.assertLen(exps, 2)

    exps = george.find_exp(all_exps, level=30)
    self.assertLen(exps, 1)
    self.assertEqual(exps[0].level, 30)

    exps = george.find_exp(all_exps, channel=1)
    self.assertLen(exps, 2)

    exp = george.find_noise_exp(all_exps)
    self.assertIsInstance(exp, george.MouseExp)
    self.assertEqual(exp.level, 10)

    exps = george.group_experiments(all_exps)
    self.assertIsInstance(exps, dict)
    self.assertLen(exps, 1)
    self.assertLen(exps['control1_pre4'], 2)

    all_exps[0].basename = ABRGeorgeTests.csv_name_3
    exps = george.group_experiments(all_exps)
    self.assertLen(exps, 2)
    self.assertLen(exps['control1_pre4'], 1) 
    self.assertLen(exps['control2_pre4'], 1) 

    (all_data, exp_freqs, exp_levels, 
     exp_channels) = george.gather_all_trial_data(all_exps)
    self.assertEqual(all_data.shape, (1, 2, 1, 10, 4))
    self.assertLen(exp_freqs, 1)
    self.assertLen(exp_levels, 2)
    self.assertLen(exp_channels, 1)

  def test_shuffle(self):
    data1 = np.reshape(np.arange(10), (-1, 1))
    data2 = np.reshape(np.arange(10) + 100, (-1, 1))
    data = np.concatenate([data1, data2], axis=1)
    self.assertEqual(data.shape, (10, 2))
    data = george.shuffle_data(data)

    self.assertLess(np.max(data[:, 0]), 10) # First row
    self.assertLen(set(data[:, 0]), 10) # Make sure nothing in lost
    self.assertNotEqual(list(data[:, 0]), list(data1[:, 0]))

    self.assertGreater(np.min(data[:, 1]), 99) # Second row
    self.assertLen(set(data[:, 1]), 10) # Make sure nothing is lost
    self.assertNotEqual(list(data[:, 1]), list(data2[:, 0]))

  def test_preprocess(self):
    """Test the various pieces of the preprocessing filter."""
    data = np.array([[1, 2], [3, 10], [5, 12]]) # 3 times in 2 trials.
    filtered = george.preprocess_mouse_data(data, remove_dc=True, 
                                            remove_artifacts=False)
    np.testing.assert_allclose(np.mean(filtered, axis=0), 
                               0.0) # Time is along the 1st axis
    
    filtered = george.preprocess_mouse_data(data, remove_dc=False, 
                                            remove_artifacts=True)
    self.assertEqual(filtered.shape, (3, 1))
    np.testing.assert_equal(filtered, [[1], [3], [5]])

    num_points = 4096
    data = np.zeros((num_points, 1))
    data[100, 0] = 1.0 # Results are bad if first point (due to filtfilt?)
    low_cutoff = 2000
    high_cutoff = 4000
    filtered = george.preprocess_mouse_data(data, remove_dc=False, 
                                            remove_artifacts=False,
                                            bandpass_filter=True,
                                            low_filter=low_cutoff,
                                            high_filter=high_cutoff)
    spectrum = 20*np.log10(np.abs(np.fft.fft(filtered, axis=0)))
    freqs = np.fft.fftfreq(num_points, d=1/george.mouse_sample_rate)
    plt.clf()
    plt.plot(freqs[:num_points//2-1], spectrum[:num_points//2-1])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Response (dB)')
    plt.title('Preprocessing Spectrum'
              f' (BP between {low_cutoff} and {high_cutoff}Hz)')
    plt.xlim(0, 8000)
    plt.savefig('test_preprocess_spectrum.png')

    passband_freqs = freqs[np.where(spectrum[:num_points//2] > -6.02)[0]]
    self.assertAlmostEqual(np.min(passband_freqs), low_cutoff, delta=10)
    self.assertAlmostEqual(np.max(passband_freqs), high_cutoff, delta=50)


  def XXtest_dprime(self):
    data = []
    num_points = 40
    num_trials = 10  # Small number of trials to see the difference
    num_exps = 500
    rng = np.random.default_rng(seed=0)

    t = np.arange(num_points)/num_points
    signal = np.sin(t*np.pi*2) * (1-t)  # Create (linearly) decaying sinusoid

    dprimes_self = []
    dprimes_wo_self = []
    dprimes_theory = []
    dprimes_noise = []
    for _ in range(num_exps):
      # First test with coherent signals.
      noise = rng.normal(scale=1, size=(num_points, num_trials))
      data = np.reshape(signal, (-1, 1)) + noise
      self.assertEqual(data.shape, (num_points, num_trials))

      dprimes_self.append(george.calculate_cov_dprime(data, 
                                                      with_self_similar=True))
      dprimes_wo_self.append(george.calculate_cov_dprime(data, 
                                                         with_self_similar=False))
      dprimes_theory.append(george.calculate_cov_dprime(data, 
                                                        with_self_similar=True, 
                                                        theoretical_model=signal))

      # Then test with incoherent signals.
      dprimes_noise.append(george.calculate_cov_dprime(noise, 
                                                       with_self_similar=False))
    dprime_self = np.mean(dprimes_self)
    dprime_wo_self = np.mean(dprimes_wo_self)
    dprime_theory = np.mean(dprimes_theory)
    dprime_noise = np.mean(dprimes_noise)
    print(f'test_dprime: dprime_self={dprime_self}, '
          f'dprime_wo_self={dprime_wo_self}, ')
    print(f'test_dprime: dprime_theory={dprime_theory}, '
          f'dprime_noise={dprime_noise}, ')

    # We expect to see 
    #   dprime_self > dprime_theory > dprime_wo_self > dprime_noise

    self.assertGreater(dprime_self, 2.5)
    self.assertGreater(dprime_self, dprime_theory)
    self.assertGreater(dprime_theory, dprime_wo_self)
    self.assertLess(dprime_noise, 1)  

  def test_dprime_sets(self):
    rng = np.random.default_rng(seed=0)

    def create_data(num_waveforms=5, num_points=20, level=10, noise=0.1):
      data = []
      for i in range(5):
        data.append(np.reshape(np.arange(num_points)/num_points*np.pi*2 + 
                              rng.normal(scale=noise, 
                                         size=num_points), (-1, 1)))
      data = np.concatenate(data, axis=1)
      return george.MouseExp('filename', '20230802_cnqx1_pre-0-30-2-1.csv', 
                             16000, level, channel=1,
                             sgi=1, single_trials=data)
    
    all_exps = []
    for level in [10, 30, 50]:
      all_exps.append(create_data(level=level, noise=1.0/level))

    # all_dprimes = george.calculate_all_summaries(all_exps)
    # self.assertLen(all_dprimes, 1)
    # self.assertIsInstance(all_dprimes['cnqx1_pre'], george.DPrimeResult)
    # dp = all_dprimes['cnqx1_pre'] 
    # self.assertEqual(dp.freqs, [16000])
    # self.assertEqual(dp.levels, [10, 30, 50])
    # self.assertEqual(dp.channels, [1])
    # self.assertEqual(dp.cov_dprimes.shape, (1, 3, 1))
    # george.plot_dprimes(dp)
    # plt.savefig('/tmp/dprime_plot.png')

  def test_caching(self):
    os.makedirs(ABRGeorgeTests.cache_dir, exist_ok=True)
    pickle_name= 'cache.pkl'
    try:
      os.remove(os.path.join(ABRGeorgeTests.cache_dir, pickle_name))
    except OSError as error:
      pass
    george.cache_all_mouse_dir(ABRGeorgeTests.csv_dir, 
                               ABRGeorgeTests.cache_dir, 
                               waveform_pickle_name=pickle_name)
    all_trials = george.load_waveform_cache(ABRGeorgeTests.cache_dir, pickle_name)
    self.assertLen(all_trials, 2)
    george.summarize_all_data([ABRGeorgeTests.cache_dir], pickle_name)
 

  def test_synthetic_rms(self):
    """Test SNR vs. window size calculation by using a synthetic ABR and 
    making sure that the peak response (when considered one short window at
    a time) is in the right place."""
    num_trials = 1024
    window_step = 50
    expected_peak = 2.3
    expected_delta = 1.0   # ToDo(malcolm) Need to tighten this bound

    synthetic_test_stack = metrics.create_synthetic_stack(noise_level=.1, 
                                                          num_trials=num_trials)
    synthetic_test_stack = np.expand_dims(synthetic_test_stack, [0, 2])
    self.assertEqual(synthetic_test_stack.shape, (1, 2, 1, 1952, num_trials))

    (synthetic_snrs, 
     synthetic_time_windows) = george.snr_vs_window_size(
       synthetic_test_stack, window_step=window_step, freq_index=0)
    peak_time = synthetic_time_windows[np.argmax(np.diagonal(
      synthetic_snrs, offset=1))]/george.mouse_sample_rate*1000

    plt.clf()
    plt.plot(synthetic_time_windows[:-1]/george.mouse_sample_rate*1000, 
             np.diagonal(synthetic_snrs, offset=1))
    plt.title('Instanenous RMS Signal Level for Synthetic ABR')
    plt.axvline(peak_time, ls=':')
    plt.axvline(expected_peak - expected_delta, ls='--')
    plt.axvline(expected_peak + expected_delta, ls='--')
    plt.xlabel('Time (ms)');
    plt.savefig('test_synthetic_rms.png')

    self.assertEqual(len(synthetic_time_windows), 
                     int(synthetic_test_stack.shape[3]/window_step) + 1)
    self.assertAlmostEqual(peak_time, expected_peak, delta=expected_delta)


class FittingTests(absltest.TestCase):
  def test_polynomial(self):
    x = np.array([1, 2, 3, 4])
    y = x**2 + 3*x
    pp = george.PositivePolynomial()
    pp.fit(x, y)
    new_x = 3.5
    self.assertAlmostEqual(pp.eval(3.5), new_x**2 + 3*new_x, delta=1e-4)

    pp = george.PositivePolynomial(semilogx=True)
    x = [1, 10,      1000, 10000]
    y = [1,  2,      4,    5]
    pp.fit(x, y)
    self.assertAlmostEqual(pp.eval(100), 3, delta=1e-4)


  def test_binlinear(self):
    x = [1, 2, 3, 4]
    y = [2, 4, 4, 5]
    bp = george.BilinearInterpolation()
    bp.fit(x, y)
    self.assertAlmostEqual(bp.eval(1), 2)
    self.assertAlmostEqual(bp.eval(1.5), 3)
    self.assertAlmostEqual(bp.eval(2.5), 4)
    self.assertAlmostEqual(bp.eval(4), 5)
    # Test extrapolation
    self.assertAlmostEqual(bp.eval(0), 0)
    self.assertAlmostEqual(bp.eval(5), 6)
    # Threshold
    self.assertAlmostEqual(bp.threshold(0), 1)
    self.assertAlmostEqual(bp.threshold(3), 1.5)
    self.assertAlmostEqual(bp.threshold(4), 2)
    self.assertAlmostEqual(bp.threshold(4.5), 3.5)
    self.assertAlmostEqual(bp.threshold(5.5), 4)

    bp = george.BilinearInterpolation(semilogx=True)
    x = [1, 10,      1000]
    y = [1,  2,      4]
    bp.fit(x, y)
    self.assertAlmostEqual(bp.eval(100), 3)

  def test_short_data(self):
    x = [1]
    y = [2, 3]
    bp = george.BilinearInterpolation()
    self.assertRaises(ValueError, bp.fit, x, y)

    y = [2]
    bp.fit(x, y)
    self.assertAlmostEqual(bp.eval(5), y[0])  # Only one point to fit, return it

  def test_dprime_calculation(self):
    h1_mean = 10
    h1_std = 3
    h2_mean = 1
    h2_std = 2

    num = 1000000
    h1_data = np.random.randn(num)*h1_std + h1_mean # Should be the bigger data
    h2_data = np.random.randn(num)*h2_std + h2_mean
                         
    dprime = george.calculate_dprime(h1_data, h2_data, geometric_mean = False)
    self.assertAlmostEqual(dprime, 
                           (h1_mean - h2_mean)/(math.sqrt((h1_std**2 + 
                                                           h2_std**2)/2)), 
                           delta=0.01)

    dprime = george.calculate_dprime(h1_data, h2_data, geometric_mean = True)
    self.assertAlmostEqual(dprime, (h1_mean - h2_mean)/(np.sqrt(h1_std*h2_std)), 
                           delta=0.01)


class EnsembleTests(absltest.TestCase):
  def create_experiments(self, count, freq, level, channel):
    exps = []
    mouse_sample_rate = 24414.0
    num_points = int(mouse_sample_rate/50) # 20ms
    # Want array of num_waveform samples x num_trials
    t = np.arange(num_points).reshape((-1, 1))/mouse_sample_rate
    t = np.concatenate(count*[t], axis=1)
    data = (level**2)*np.sin(2*np.pi*t*freq) + 4*np.random.standard_normal(t.shape)
    # print(f'freq={freq}, level={level}, channel={channel} RMS is:',
    #       np.sqrt(np.mean(data**2)))

    exp = george.MouseExp(filename=f'Filename {level}',
                          basename=f'Base {level}',
                          sgi=level,
                          channel=channel,
                          freq=freq,
                          level=level,
                          description=f'Test {level}',
                          single_trials=data)
    return [exp]

  def test_rms_calculation(self):
    # num_samples x num_trials
    data = np.array([[1, 2], [1, 2], [1, 2], [1, 2]])
    np.testing.assert_almost_equal(george.calculate_rms(data),
                                   np.array([1, 2]))

  def XXtest_measures(self):
    """Create fake data, and see if the waveform summaries work right."""
    all_exps = []
    freqs = [500] # Preprocessing filter goes from 200Hz to 1kHz
    levels = [1, 2, 3, 4]
    channels = [1, 2]
    for freq in freqs:
      for level in levels:
        for channel in channels:
          # Create a lot of experiments so histograms are smooth.
          all_exps += self.create_experiments(1000, freq, level, channel)

    # First calculate (and plot) the Covariance measure
    plt.clf()
    res = george.DPrimeResult(*george.calculate_waveform_summaries( 
      all_exps, True, debug_freq=freq, debug_levels=levels, debug_channel=1))
    plt.savefig('test_ensemble_cov_dprime.png')
    dprimes_result = {'test': res}
    print('test_measures res are:', res)
    
    # Result is indexed by frequency, level, and channel, d' goes up with level
    self.assertLess(res.cov_dprimes[0, 0, 0], 20.0)
    self.assertGreater(res.cov_dprimes[0, 3, 0], 62.0)
    np.testing.assert_array_less(res.cov_dprimes[0, :-1, 0], 
                                 res.cov_dprimes[0, 1:, 0])

    # Second, calculate again, but plot the RMS measures this time.
    plt.clf()
    res = george.DPrimeResult(*george.calculate_waveform_summaries( 
       all_exps, False, debug_freq=freq, debug_levels=levels, debug_channel=1))
    plt.savefig('test_ensemble_rms_dprime.png')
    np.testing.assert_array_less(res.rms_dprimes[0, :-1, 0], 
                                 res.rms_dprimes[0, 1:, 0])
    np.testing.assert_array_less(res.rms_of_total[0, :-1, 0], 
                                 res.rms_of_total[0, 1:, 0])
                    
    # Expectation is sqrt(level**2 * RMS(sin) + RMS(noise))
    expectations = np.sqrt((np.arange(1, 5)**2*np.sqrt(1/2.0))**2 + 16)

    print('Cov dprimes:', res.cov_dprimes)
    print('RMS of Total Signal:', res.rms_of_total)
    print('RMS dprimes:', res.rms_dprimes)
    np.testing.assert_allclose(res.rms_of_total[0, :, 0], expectations, 
                               rtol=.01)
  
    plt.clf()
    res.add_threshold(dp_criteria=20, fit_method='bilinear', plot=True)
    plt.savefig('test_ensemble_threshold.png')
    print(res)
    self.assertAlmostEqual(res.cov_spl_threshold[0][0], 2.183, delta=0.045)
    np.testing.assert_array_less(res.cov_smooth_dprimes[0, :-1, 0], 
                                 res.cov_smooth_dprimes[0, 1:, 0])


 
if __name__ == "__main__":
  absltest.main()