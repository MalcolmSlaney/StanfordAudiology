import sys

from absl.testing import absltest
import numpy as np
import pandas as pd
import traitlets

import clusters


class ClusterTests(absltest.TestCase):
  def test_euclidean_distance(self):
    a = np.array([0, 0])
    b = np.array([3, 4])
    c = clusters.euclidean_distance(a, b)
    self.assertAlmostEqual(c, 5, delta=1e-5)

  def test_convert_spaces(self):
    res = clusters.ConvertSpaces(['foo ', ' bar'])
    self.assertEqual(res, ['foo_', '_bar'])

  def test_make_pandas(self):
    test_data = [['foo ', ' bar'],
                 ['1', '2'],
                 ['3', '4']]
    df = clusters.MakePandas(test_data)
    self.assertListEqual(list(df.columns), ['foo_', '_bar'])
    np.testing.assert_equal(df.values, np.array([[1, 2], [3, 4]]))

  def test_rename_duplicates(self):
    input = ['foo', 'bar', 'baz', 'bar']
    res = clusters.RenameDuplicateColumns(input, 'bar')
    self.assertListEqual(res, ['foo', 'bar', 'baz', 'barDupe'])


if __name__=="__main__": 
  absltest.main()