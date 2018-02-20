import functools

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.signal.python.ops import window_ops

__author__ = 'Andres'

sampling_rate = 44000
freq = 440
countOfCycles = 4
_time = tf.range(0, 1024 / sampling_rate, 1 / sampling_rate, dtype=tf.float32)
firstSignal = tf.sin(2 * 3.14159 * freq * _time)

with tf.name_scope('Energy_Spectogram'):
    fft_frame_length = 128
    fft_frame_step = 32
    window_fn = functools.partial(window_ops.hann_window, periodic=True)
    firstSignal = tf.concat([tf.zeros(fft_frame_length-fft_frame_step), firstSignal, tf.zeros(fft_frame_length-fft_frame_step)], axis=0)
    stft = tf.contrib.signal.stft(signals=firstSignal, frame_length=fft_frame_length, frame_step=fft_frame_step,
                                  fft_length=fft_frame_length, window_fn=window_fn)
    istft = tf.contrib.signal.inverse_stft(stfts=stft, frame_length=fft_frame_length, frame_step=fft_frame_step,
    window_fn=tf.contrib.signal.inverse_stft_window_fn(fft_frame_step,
                                           forward_window_fn=window_fn))

with tf.Session() as sess:
    original, reconstructed = sess.run([firstSignal, istft])

def _pavlovs_SNR(y_orig, y_inp):
    norm_y_orig = np.linalg.norm(y_orig) + 1e-10
    norm_y_orig_minus_y_inp = np.linalg.norm(y_orig - y_inp)
    return 10 * np.log10((abs(norm_y_orig ** 2)) / abs((norm_y_orig_minus_y_inp ** 2)))

print(_pavlovs_SNR(original, reconstructed))

plt.plot(original)
plt.plot(reconstructed)
plt.show()
