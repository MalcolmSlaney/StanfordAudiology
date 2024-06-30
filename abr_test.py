from absl.testing import absltest
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


if __name__=="__main__": 
  absltest.main()