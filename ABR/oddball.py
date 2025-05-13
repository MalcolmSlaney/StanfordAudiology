import json
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.io import wavfile

from typing import List


def create_sequence(count: int, normal_frac:float =.9) -> List[bool]:
  """Create the oddball sequence.  Return a list of booleans, with True for
  normal and False for deviant.
  """
  normal = np.random.randn(count) < normal_frac
  consecutive = np.logical_and(normal[:-1] == False, normal[1:] == False)
  consecutive = np.where(consecutive)[0]
  if len(consecutive):
    normal[consecutive] = True
  return normal


def choose_block(s: bool, fs: float, 
                 normal_beep: NDArray, deviant_beep: NDArray,
                 silence: NDArray, normal_sense: bool = True):
  silence_ms = np.random.uniform(700, 899)
  silence_samples = int(silence_ms/1000*fs)
  jittered_silence = silence[0:silence_samples]

  if s == normal_sense:
    return [normal_beep, jittered_silence]
  else:
    return [deviant_beep, jittered_silence]


def create_block(count = 400, sense=True):
  isi = 900.0 # ms
  stimulus_length = 200.0 # ms

  normal_frac = 0.9
  deviant_frac = 0.1

  standard_freq = 500
  deviant_freq = 650

  block_length = 400
  block_count = 2

  fs = 22050
  t = np.linspace(0, stimulus_length/1000, int(fs*stimulus_length/1000))

  taper = 0.010 # Seconds
  window = (1 - np.cos(t[t<taper]/taper*np.pi))/2
  window = np.concatenate((window, 
                          np.ones(t.shape[0] - 2*window.shape[0]),
                          np.flip(window)))

  normal_beep = np.sin(2*np.pi*standard_freq*t) * window
  deviant_beep = np.sin(2*np.pi*deviant_freq*t) * window
  silence = np.zeros(int(isi/1000*fs))

  s = create_sequence(count)
  one_block = [choose_block(i, sense) for i in s]
  res = list(sum(one_block, []))
  return np.concatenate(res), s, [r.shape[0] for r in res]

