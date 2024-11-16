import glob
import json
import numpy as np
import os
import pandas as pd
from typing import List, Optional

from absl import app
from absl import flags

import matplotlib.pyplot as plt
import zarr
from cftsdata import abr


basedir = 'drive/Shareddrives/StanfordAudiology/ABRPresto/ABRpresto data'


def read_experiments(path):
  load_options = {}

  if not os.path.exists(os.path.join(path, 'erp_metadata.csv')):
    raise ValueError(f'{path} does not contain ABR data')
  fh = abr.load(path)
  epochs = fh.get_epochs_filtered(**load_options)
  return epochs.sort_index()

  # for freq, freq_df in epochs.groupby('frequency'):
  #   yield freq, freq_df


def get_unique_levels(freq_df):
  levels = []
  for f, l, p, t in freq_df.index:
    levels.append(l)
  return list(set(levels))

def get_unique_freqs(freq_df):
  freqs = []
  for f, l, p, t in freq_df.index:
    freqs.append(f)
  return list(set(freqs))


def get_unique_polarities(freq_df):
  polarities = []
  for f, l, p, t in freq_df.index:
    polarities.append(p)
  return list(set(polarities))


def get_one_df(freq_df, freq, level, polarity='both'):
  if polarity == 'both':
    pos = freq_df.loc[(freq, level, 1)].to_numpy()
    neg = freq_df.loc[(freq, level, -1)].to_numpy()
    num = min(pos.shape[0], neg.shape[0])
    return pos[:num, :] + neg[:num, :]
  else:
    return freq_df.loc[(freq, level, polarity)].to_numpy()


def calculate_dprime(data, exp_axis=0, plot_histogram=False,
                     use_noise=True):
  """Calculate d' for an array of ABR Responses.

  Args:
    data: An array of num_experiments, each with num_samples.  The exp_axis
      argument specifies which dimension represents the experiments.
    exp_axis: Which dimension corresponds to experiment, and the other is time
    plot_histogram: Whether plot the histogram
    use_noise: Randomize the data so it is noise, and use RMS of noise and
      signal standard deviations.

  Returns:
    A scalar d' estimate of the covariance model
  """
  def model_response(data, model) -> List[float]:
    """Return covariance for each trial."""
    return

  if isinstance(data, pd.DataFrame):
    data = data.to_numpy()

  model = np.mean(data, axis=exp_axis, keepdims=True)
  model_resp = np.sum(data * model, axis=1-exp_axis)
  model_mean = np.mean(model_resp)
  model_std = np.std(model_resp)
  if plot_histogram:
    plt.hist(model_resp, 20)

  if use_noise:
    if exp_axis == 0:
      noise = np.copy(data.T)
      np.random.shuffle(noise) # Shuffle in time (axis 1)
      noise = noise.T
    else:
      noise = np.copy(data)
      np.random.shuffle(noise) # Shuffle in time (axis 0)
    noise_resp = np.sum(noise * model, axis=1-exp_axis)
    noise_mean = np.mean(noise_resp)
    noise_std = np.std(noise_resp)
    if plot_histogram:
      plt.hist(noise_resp, 20)
    std = np.sqrt((model_std**2 + noise_std**2)/2)
    dprime = (model_mean - noise_mean) / std
  else:
    dprime = model_mean / model_std
  return dprime


def summarize_exp(freq_df, make_plots=False):
  """

  Args:
    freq_df:

  Returns:

  """
  freqs = sorted(get_unique_freqs(freq_df))
  levels = sorted(get_unique_levels(freq_df))
  freq_results = {}
  for freq in freqs:
    level_dprime = {}
    for level in levels:
      level_dprime[level] = calculate_dprime(get_one_df(freq_df, freq, level, 'both'),
                                           use_noise=True)
    if make_plots:
      plt.plot(levels, level_dprime.values(), label=freq)
    freq_results[freq] = level_dprime
  if make_plots:
    plt.legend()
    plt.xlabel('SPL (dB)')
    plt.ylabel('d\'')
  return freq_results


def compute_all_dprimes(all_mice, all_exp_dprimes = {}):
  num_computed = 0
  for one_exp in all_mice:
    just_mouse_name = one_exp.replace(basedir + '/', '')
    if just_mouse_name in all_exp_dprimes:
      continue
    # plt.figure()
    print(one_exp)
    epochs = read_experiments(one_exp)
    all_exp_dprimes[one_exp.replace(basedir + '/', '')] = summarize_exp(epochs)
    plt.title(one_exp.replace(basedir + '/', ''))
    num_computed += 1
    if num_computed % 50 == 0:
      print(f'Saving {len(all_exp_dprimes)} records')
      with open('drive/Shareddrives/StanfordAudiology/ABRPresto/all_exp_dprimes.json', 'w') as f:
        json.dump(all_exp_dprimes, f)
  print('all done')
  return all_exp_dprimes



FLAGS = flags.FLAGS
flags.DEFINE_string('basedir',
                    'drive/Shareddrives/StanfordAudiology/ABRPresto/',
                    'Base directory to find the ABRPresto mouse data')
flags.DEFINE_string('json', 'all_exp_dprimes.json', 
                    'Where to read and store the results cache file')

def main():
  one_exp = glob.glob(os.path.join(FLAGS.basedir, 'ABRpresto data', 
                                   '*350*timepoint0*'))[0]
  all_mice = glob.glob(os.path.join(basedir, 'Mouse*abr*'))

  all_exp_dprimes = {}

  with open(os.path.join(FLAGS.basedir, 'all_exp_dprimes.json'), 'r') as f:
    all_exp_dprimes = json.load(f)
  print(f'Loaded {len(all_exp_dprimes)} mice experiments')

  all_exp_dprimes = compute_all_dprimes(all_mice, all_exp_dprimes)

  with open(os.path.join(FLAGS.basedir, FLAGS.json), 'w') as f:
    json.dump(all_exp_dprimes, f)




if __name__ == '__main__':
  app.run(main)