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
    dprime = george.calculate_dprime(data)
    self.assertGreater(dprime, 15)

    # Then test with incoherent signals.
    data = []
    for i in range(5):
      data.append(np.reshape(np.arange(num_points)/num_points*np.pi*i + 
                             rng.normal(scale=1,size=num_points), (-1, 1)))
    data = np.concatenate(data, axis=1)
    dprime = george.calculate_dprime(data)
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

    all_dprimes = george.calculate_all_dprimes(all_exps)
    self.assertLen(all_dprimes, 1)
    self.assertIsInstance(all_dprimes['cnqx1_pre'], george.DPrimeResult)
    dp = all_dprimes['cnqx1_pre'] 
    self.assertEqual(dp.freqs, [16000])
    self.assertEqual(dp.levels, [10, 30, 50])
    self.assertEqual(dp.channels, [1])
    self.assertEqual(dp.dprimes.shape, (1, 3, 1))
    george.plot_dprimes(dp)
    plt.savefig('/tmp/dprime_plot.png')

  def test_caching(self):
    basedir = '/tmp'
    pickle_name= 'cache.pkl'
    try:
      os.remove(os.path.join(basedir, pickle_name))
    except OSError as error:
      pass
    all_trials = george.cache_waveform_data(basedir,
                                            waveform_pickle_name=pickle_name, 
                                            load_data=True)
    self.assertLen(all_trials, 2)
    george.summarize_all_data([basedir], pickle_name)
 
if __name__ == "__main__":
  absltest.main()