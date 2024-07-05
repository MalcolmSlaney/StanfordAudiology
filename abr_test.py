from absl.testing import absltest
import matplotlib.pyplot as plt
import numpy as np

import abr


class TestProcessingModules(absltest.TestCase):
  def test_offset_removal(self):
    simulated_data = np.random.rand(500, 6000)
    filtered_data = abr.remove_offset(simulated_data)
    np.testing.assert_almost_equal(np.mean(filtered_data, axis=0), 
                                   np.zeros(6000))

  def test_bypassing_artifact_rejection(self):
    simulated_data = np.random.rand(500, 6000)
    # rejecting artifacts above the 100 percentile is the same as applying no filtering
    filtered_data = abr.reject_artifacts(simulated_data, 
                                         variance_percentile=100, 
                                         max_magnitude_percentile=100, 
                                         rms_percentile=100)
    self.assertTrue(np.all(simulated_data == filtered_data))

  def test_artifact_rejection_90_percentile(self):
    simulated_data = np.random.rand(500, 6000)
    filtered_data = abr.reject_artifacts(simulated_data, variance_percentile=90, 
                                         max_magnitude_percentile=90, 
                                         rms_percentile=90)

    variance_threshold = np.percentile(np.var(simulated_data, axis=0), 90)
    max_magnitude_threshold = np.percentile(np.max(np.abs(simulated_data), 
                                                   axis=0), 90)
    rms_threshold = np.percentile(abr.compute_rms(simulated_data), 90)

    self.assertFalse(np.any(np.var(filtered_data, axis=0) > variance_threshold))
    self.assertFalse(np.any(np.max(np.abs(filtered_data), axis=0) > 
                            max_magnitude_threshold))
    self.assertFalse(np.any(abr.compute_energy(filtered_data) / 6000 > 
                            rms_threshold))

  def test_filter(self):
    def evaluate_filter(lower_cutoff: float, upper_cutoff: float, 
                    fs: float = 16000, order: int = 8):
      n = 16384
      impulse = np.zeros(n)
      impulse[0] = 1.0
      
      y = abr.butterworth_filter(impulse, lower_cutoff, upper_cutoff, 
                                 fs, order=order)
      
      y += np.random.randn(n)*1e-10  # Add noise so next log doesn't get upset.
      spectrum = 20*np.log10(np.abs(np.fft.fft(y)))
      freqs = np.arange(n)/float(n)*fs
      with open(f'test_filter_spectrum_{lower_cutoff}_{upper_cutoff}_'
                f'{order}.png', 'wb') as fp:
        plt.semilogx(freqs, spectrum)
        plt.xlim(1, fs/2)
        plt.ylim(-13, 1)
        plt.xlabel('Frequency (Hx)')
        plt.ylabel('Filter Response (dB)')
        plt.title('EEG Filter');
        plt.savefig(fp)

      spectrum[0] = spectrum[1]
      passband_freqs = freqs[np.where(spectrum[:n//2] > -3.02)[0]]
      return np.min(passband_freqs), np.max(passband_freqs)
    
    # Bandpass filter
    low_cutoff, high_cutoff = evaluate_filter(500, 1000, 4000)
    self.assertAlmostEqual(low_cutoff, 500, delta=1)
    self.assertAlmostEqual(high_cutoff, 1000, delta=1)

    # Lowpass filter
    low_cutoff, high_cutoff = evaluate_filter(0, 1000, 4000)
    self.assertAlmostEqual(low_cutoff, 0, delta=1)
    self.assertAlmostEqual(high_cutoff, 1000, delta=1)

    # Highpass filter
    low_cutoff, high_cutoff = evaluate_filter(1000, 2000, 4000)
    self.assertAlmostEqual(low_cutoff, 1000, delta=1)
    self.assertAlmostEqual(high_cutoff, 2000, delta=1)

                         
if __name__=="__main__": 
  absltest.main()