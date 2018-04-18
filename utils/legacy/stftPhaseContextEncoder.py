import numpy as np
import tensorflow as tf

from utils.legacy.stftGapContextEncoder import StftGapContextEncoder

__author__ = 'Andres'


class StftPhaseContextEncoder(StftGapContextEncoder):
    def _loss_graph(self):
        with tf.variable_scope("Loss"):
            gap_stft = self._target_model.output()

            abs_stft = tf.reshape(gap_stft[:, :, :, 0], (self._batch_size, 11, 257, 1))
            target_angle = abs_stft * tf.reshape(gap_stft[:, :, :, 1], (self._batch_size, 11, 257, 1))

            norm_orig = self._squaredEuclideanNorm(target_angle, onAxis=[1, 2, 3])
            norm_orig_summary = tf.summary.scalar("norm_orig", tf.reduce_min(norm_orig))

            error = target_angle - (self._reconstructed_input_data * abs_stft)
            error_per_example = tf.reduce_sum(tf.square(error), axis=[1, 2, 3])

            reconstruction_loss = 0.5 * tf.reduce_sum(error_per_example * (1 + 5 / (norm_orig + 1e-2)))

            rec_loss_summary = tf.summary.scalar("reconstruction_loss", reconstruction_loss)

            trainable_vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]) * 1e-2
            l2_loss_summary = tf.summary.scalar("lossL2", lossL2)

            total_loss = tf.add_n([reconstruction_loss, lossL2])
            total_loss_summary = tf.summary.scalar("total_loss", total_loss)

            self._lossSummaries = tf.summary.merge(
                [rec_loss_summary, l2_loss_summary, norm_orig_summary, total_loss_summary])

            return total_loss

    def _evaluateValidSNR(self, summaries_dict, validReader, evalWriter, writer, sess, step):
        reconstructed, out_gaps = self._reconstruct(sess, validReader, max_steps=8)
        reconstructed = np.reshape(reconstructed, (self._batch_size*8, 11, 257, 1))
        step_valid_SNR = evalWriter.evaluateImages(reconstructed, np.reshape(out_gaps[:, :, :, 1], (self._batch_size*8, 11, 257, 1)), self._initial_model_num + step)
        validSNRSummaryToWrite = sess.run(summaries_dict['valid_SNR_summary'],
                                          feed_dict={summaries_dict['valid_SNR']: step_valid_SNR})
        writer.add_summary(validSNRSummaryToWrite, self._initial_model_num + step)