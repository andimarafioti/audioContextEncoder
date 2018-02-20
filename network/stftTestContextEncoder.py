import tensorflow as tf
import numpy as np
from network.contextEncoder import ContextEncoderNetwork
from utils.strechableNumpyArray import StrechableNumpyArray

__author__ = 'Andres'


class StftTestContextEncoder(ContextEncoderNetwork):
    def _loss_graph(self):
        with tf.variable_scope("Loss"):
            fft_frame_length = 512
            fft_frame_step = 128
            stft = tf.contrib.signal.stft(signals=self.gap_data, frame_length=fft_frame_length,
                                          frame_step=fft_frame_step)
            mag_stft = tf.abs(stft)

            # fft_unique_bins = fft_frame_length // 2 + 1  # 257
            # num_ffts = int((self._gap_length - fft_frame_length) / fft_frame_step) + 1  # 5

            # norm_orig = self.euclideanNorm(self.gap_data) / 5

            error = mag_stft - self._reconstructed_input_data
            reconstruction_loss = 0.5 * tf.reduce_sum(tf.reduce_sum(tf.square(error), axis=1))  # * (1 + 1 / norm_orig))
            tf.summary.scalar("reconstruction_loss", reconstruction_loss)

            trainable_vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]) * 1e-2
            tf.summary.scalar("lossL2", lossL2)

            total_loss = tf.add_n([reconstruction_loss, lossL2])
            tf.summary.scalar("total_loss", total_loss)

            return total_loss

    def _reconstruct(self, sess, data_reader, max_steps):
        data_reader.start()
        reconstructed = StrechableNumpyArray()
        out_gaps = StrechableNumpyArray()
        for batch_num in range(max_steps):
            try:
                sides, gaps = data_reader.dataOperation(session=sess)
            except StopIteration:
                print(batch_num)
                print("rec End of queue!")
                break
            fft_frame_length = 512
            fft_frame_step = 128
            stft = tf.contrib.signal.stft(signals=gaps, frame_length=fft_frame_length,
                                          frame_step=fft_frame_step)
            mag_stft = tf.abs(stft)

            feed_dict = {self._model.input(): sides, self.gap_data: gaps}
            reconstructed_input, original = sess.run([self._reconstructed_input_data, mag_stft], feed_dict=feed_dict)
            out_gaps.append(np.reshape(original, (-1)))
            reconstructed.append(np.reshape(reconstructed_input, (-1)))
        reconstructed = reconstructed.finalize()
        reconstructed = np.reshape(reconstructed, (-1, 5, 257))

        out_gaps = out_gaps.finalize()
        out_gaps = np.reshape(out_gaps, (-1, 5, 257))
        data_reader.finish()

        return reconstructed, out_gaps

