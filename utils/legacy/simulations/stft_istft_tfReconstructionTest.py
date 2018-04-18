import functools

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.contrib.signal.python.ops import window_ops

__author__ = 'Andres'


s = tf.Session()

sampling_rate = 44000
freq = 210
countOfCycles = 4
_time = tf.range(0, 1024 / sampling_rate, 1 / sampling_rate, dtype=tf.float32)
firstSignal = tf.sin(2 * 3.14159 * freq * _time)

fft_frame_length = 512
fft_frame_step = 128
window_fn = functools.partial(window_ops.hann_window, periodic=True)
inverse_window = tf.contrib.signal.inverse_stft_window_fn(fft_frame_step,
                                           forward_window_fn=window_fn)

firstSignal = tf.concat([tf.zeros(fft_frame_length-fft_frame_step), firstSignal, tf.zeros(fft_frame_length-fft_frame_step)], axis=0)
s.run(tf.initialize_all_variables())
stft = tf.contrib.signal.stft(signals=firstSignal, frame_length=fft_frame_length, frame_step=fft_frame_step,
                              fft_length=fft_frame_length, window_fn=window_fn)
istft = tf.contrib.signal.inverse_stft(stfts=stft, frame_length=fft_frame_length, frame_step=fft_frame_step,
                                       window_fn=inverse_window)

stft_times = []
istft_times = []
for x in range(1):
    t = time.time()
    s.run(stft)
    stft_times.append(time.time()-t)
    print('stft took:', stft_times[-1])
    t = time.time()
    s.run(istft)
    istft_times.append(time.time()-t)
    print('istft took:', istft_times[-1])

print(stft_times)
print(istft_times)
print(np.mean(stft_times))
print(np.mean(istft_times))


with tf.Session() as sess:
    t, original, stft_t, reconstructed = sess.run([_time, firstSignal, stft, istft])

def _pavlovs_SNR(y_orig, y_inp):
    norm_y_orig = np.linalg.norm(y_orig) + 1e-10
    norm_y_orig_minus_y_inp = np.linalg.norm(y_orig - y_inp)
    return 10 * np.log10((abs(norm_y_orig ** 2)) / abs((norm_y_orig_minus_y_inp ** 2)))

print(_pavlovs_SNR(original, reconstructed))

ax1 = plt.subplot(211)
plt.plot(original)
plt.plot(reconstructed)
plt.subplot(212)
print(np.transpose(np.abs(stft_t)).shape)
plt.pcolormesh(np.transpose(np.abs(stft_t)))
plt.show()

