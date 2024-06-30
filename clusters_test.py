import os
from collections import Counter

from absl.testing import absltest
import numpy as np
import pandas as pd

import clusters


class ClusterTests(absltest.TestCase):
  def test_euclidean_distance(self):
    a = np.array([0, 0])
    b = np.array([3, 4])
    c = clusters.EuclideanDistance(a, b)
    self.assertAlmostEqual(c, 5, delta=1e-5)

  def test_convert_spaces(self):
    res = clusters.ConvertSpaces(['foo ', ' bar'])
    self.assertEqual(res, ['foo_', '_bar'])

  def test_make_pandas(self):
    test_data = [['foo ', ' bar'],
                 ['1',  '2'],
                 ['3',  '4']]
    df = clusters.MakePandas(test_data)
    self.assertListEqual(list(df.columns), ['foo_', '_bar'])
    np.testing.assert_equal(df.values, np.array([[1, 2], [3, 4]]))

  def test_rename_duplicates(self):
    input = ['foo', 'bar', 'baz', 'bar']
    res = clusters.RenameDuplicateColumns(input, 'bar')
    self.assertListEqual(res, ['foo', 'bar', 'baz', 'barDupe'])

  def test_clustering(self):
    test_data = [['foo ', ' bar'],
                 [1.0,  1.0],
                 [0.9,  1.1],
                 [2.0,  2.0],
                 [1.9,  2.1],
                 ]
    num_clusters = 2
    df = clusters.MakePandas(test_data)
    np.random.seed(0)
    kmeans = clusters.CreateKMeans(num_clusters, df)
    self.assertEqual(kmeans.n_clusters, num_clusters) # , new_column_name='predictions')
    cluster_ids = clusters.KMeansPredictions(kmeans, df)
    self.assertIsInstance(cluster_ids, np.ndarray)
    # Note, clusters could be permuted, so counting classes might be better.
    # self.assertListEqual(list(cluster_ids), [0, 0, 1, 1])
    df['predictions'] = cluster_ids

    # Make sure the labels are right by counting the results.
    counts = clusters.CountPredictions(df)  # , cluster_label=None)
    self.assertDictEqual(counts, {0: 2, 1: 2})

    # Test save and restore
    #temp_dir = os.getenv('TEMP') or os.getenv('TMP')  # For windows
    #filepath = clusters.SaveAsJson(
    #   kmeans, test_data[0], test_data[0], 2, 'clusters', temp_dir)  # For windows
    filepath = clusters.SaveAsJson(kmeans, test_data[0], test_data[0], 2, 'clusters', '/tmp/')
    new_kmeans, _, _ = clusters.LoadFromJson(filepath)
    cluster_centers = np.sort(new_kmeans.cluster_centers_, axis=0)
    np.testing.assert_allclose(cluster_centers, np.array([[0.95, 1.05],
                                                          [1.95, 2.05]]))

  def test_classification(self):
    column_names = ['R250', 'R500', 'R1000', 'R2000', 'R3000', 'R4000', 'R6000',
                    'R8000', 'L250', 'L500', 'L1000', 'L2000', 'L3000', 'L4000',
                    'L6000', 'L8000', 'RBone500', 'RBone1000', 'RBone2000', 'RBone4000',
                    'LBone500', 'LBone1000', 'LBone2000', 'LBone4000']

    # Make some random data and make sure we get the same results as before.
    np.random.seed(0)
    fake_data = np.random.uniform(0, 100, size=(20, len(column_names)))
    df = pd.DataFrame(fake_data, columns=column_names)
    df_with_classes = clusters.HLossClassifier(df)
    type_counts = Counter(df_with_classes['R_Type_HL_4freq'].to_list())
    self.assertDictEqual(type_counts, {'SNHL': 10, 'Mixed': 2,
                       'Unknown': 7, 'Conductive': 1})

    type_counts = Counter(df_with_classes['L_Type_HL_4freq'].to_list())
    self.assertDictEqual(type_counts, {'SNHL': 10, 'Mixed': 6,
                       'Unknown': 4})

    # Test bone vs. air conduction comparison
    df_clean = clusters.RemoveRowsWithBCWorseAC(df_with_classes, 10)
    # Started with 20 rows, now down to 8??
    self.assertEqual(df_clean.shape[0], 8)

  def test_age(self):
    df = pd.DataFrame([-10, 10, 50, 110], columns=['AgeAtTestDate'])
    df_clean = clusters.RemoveRowsWithBadAges(df)
    self.assertEqual(df_clean.shape[0], 2)

  def test_entropy(self):
    self.assertAlmostEqual(clusters.ComputeDiscreteEntropy([0, 0, 1, 1]), 1.0)
    self.assertAlmostEqual(clusters.ComputeDiscreteEntropy([0, 0]), 0.0)
    self.assertAlmostEqual(clusters.ComputeDiscreteEntropy([0, 1, 2, 3]), 2.0)

    self.assertAlmostEqual(
      clusters.ComputeContinuousEntropy(np.arange(0, 1, .001), 0, 0.125), 3.0)
    self.assertAlmostEqual(
      clusters.ComputeContinuousEntropy(np.arange(0, 1, .001), 0, 0.25),  2.0)


if __name__ == "__main__":
  absltest.main()
