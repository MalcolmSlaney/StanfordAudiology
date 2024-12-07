import os

from absl.testing import absltest
import abr_george as george

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
  csv_name_1 = '/tmp/20230824_control1_pre4-0-52-1-1.csv'
  csv_name_2 = '/tmp/20230824_control1_pre4-0-52-1-2.csv'
  csv_name_3 = '/tmp/20230824_control2_pre4-0-52-1-2.csv'

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
    george.cache_all_mouse_dir('/tmp')
    all_exps = george.load_waveform_cache('/tmp')
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

    num_points = 512
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
    plt.title('Preprocessing Test Filter Frequency Spectrum')
    plt.xlim(0, 8000)
    plt.savefig('test_preprocess_spectrum.png')

    passband_freqs = freqs[np.where(spectrum[:num_points//2] > -6.02)[0]]
    self.assertAlmostEqual(np.min(passband_freqs), low_cutoff, delta=10)
    self.assertAlmostEqual(np.max(passband_freqs), high_cutoff, delta=50)


  def test_dprime(self):
    data = []
    num_points = 20
    rng = np.random.default_rng(seed=0)
    # First test with coherent signals.
    for i in range(5):
      data.append(np.reshape(np.arange(num_points)/num_points*np.pi*2 + 
                             rng.normal(scale=0.1,size=num_points), (-1, 1)))
    data = np.concatenate(data, axis=1)
    dprime = george.calculate_cov_dprime(data)
    self.assertGreater(dprime, 15)

    # Then test with incoherent signals.
    data = []
    for i in range(5):
      data.append(np.reshape(np.arange(num_points)/num_points*np.pi*i + 
                             rng.normal(scale=1,size=num_points), (-1, 1)))
    data = np.concatenate(data, axis=1)
    dprime = george.calculate_cov_dprime(data)
    self.assertLess(dprime, 1)

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

    all_dprimes = george.calculate_all_summaries(all_exps)
    self.assertLen(all_dprimes, 1)
    self.assertIsInstance(all_dprimes['cnqx1_pre'], george.DPrimeResult)
    dp = all_dprimes['cnqx1_pre'] 
    self.assertEqual(dp.freqs, [16000])
    self.assertEqual(dp.levels, [10, 30, 50])
    self.assertEqual(dp.channels, [1])
    self.assertEqual(dp.cov_dprimes.shape, (1, 3, 1))
    george.plot_dprimes(dp)
    plt.savefig('/tmp/dprime_plot.png')

  def test_caching(self):
    basedir = '/tmp'
    pickle_name= 'cache.pkl'
    try:
      os.remove(os.path.join(basedir, pickle_name))
    except OSError as error:
      pass
    george.cache_all_mouse_dir(basedir, waveform_pickle_name=pickle_name)
    all_trials = george.load_waveform_cache(basedir, pickle_name)
    self.assertLen(all_trials, 2)
    george.summarize_all_data([basedir], pickle_name)
 

class FittingTests(absltest.TestCase):
  def test_polynomial(self):
    x = np.array([1, 2, 3, 4])
    y = x**2 + 3*x
    pp = george.PositivePolynomial()
    pp.fit(x, y)
    new_x = 3.5
    self.assertAlmostEqual(pp.eval(3.5), new_x**2 + 3*new_x, delta=1e-4)

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
    self.assertAlmostEqual(bp.threshold(0), 0)
    self.assertAlmostEqual(bp.threshold(3), 1.5)
    self.assertAlmostEqual(bp.threshold(4), 3)
    self.assertAlmostEqual(bp.threshold(4.5), 3.5)
    self.assertAlmostEqual(bp.threshold(5.5), 4.5)

  def test_short_data(self):
    x = [1]
    y = [2, 3]
    bp = george.BilinearInterpolation()
    self.assertRaises(ValueError, bp.fit, x, y)

    y = [2]
    bp.fit(x, y)
    self.assertAlmostEqual(bp.eval(5), y[0])  # Only one point to fit, return it

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

  def test_measures(self):
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
    (cov_dprimes, rmses, rms_dprimes, 
     all_freqs, all_levels, all_channels) = george.calculate_waveform_summaries(
       all_exps, True, debug_freq=freq, debug_levels=levels, debug_channel=1)
    plt.savefig('test_ensemble_cov_dprime.png')
    
    # Result is indexed by frequency, level, and channel, d' goes up with level
    self.assertLess(cov_dprimes[0, 0, 0], 20.0)
    self.assertGreater(cov_dprimes[0, 3, 0], 64.0)
    np.testing.assert_array_less(cov_dprimes[0, :-1, 0], cov_dprimes[0, 1:, 0])

    # Second, calculate again, but plot the RMS measures this time.
    plt.clf()
    (cov_dprimes, rmses, rms_dprimes, all_freqs, 
     all_levels, all_channels) = george.calculate_waveform_summaries(
       all_exps, False, debug_freq=freq, debug_levels=levels, debug_channel=1)
    plt.savefig('test_ensemble_rms_dprime.png')
    np.testing.assert_array_less(rms_dprimes[0, :-1, 0], rms_dprimes[0, 1:, 0])
    np.testing.assert_array_less(rmses[0, :-1, 0], rmses[0, 1:, 0])
                    
    # Expectation is sqrt(level**2 * RMS(sin) + RMS(noise))
    expectations = np.sqrt((np.arange(1, 5)**2*np.sqrt(1/2.0))**2 + 16)

    print('Cov dprimes:', cov_dprimes)
    print('RMSes:', rmses)
    print('RMS dprimes:', rms_dprimes)
    np.testing.assert_allclose(rmses[0, :, 0], expectations, rtol=.01)
  
if __name__ == "__main__":
  absltest.main()