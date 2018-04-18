import re

import numpy as np
import tensorflow as tf

from utils.legacy.contextEncoder import ContextEncoderNetwork
from utils.strechableNumpyArray import StrechableNumpyArray

__author__ = 'Andres'


class StftGapContextEncoder(ContextEncoderNetwork):
    def __init__(self, model, batch_size, target_model, window_size, gap_length, learning_rate, name):
        self._target_model = target_model
        super(StftGapContextEncoder, self).__init__(model, batch_size, window_size, gap_length, learning_rate,
                                                         name)
        self._sides = tf.placeholder(tf.float32, shape=(batch_size, self._window_size - self._gap_length), name='sides')
        self._reconstructedSignal = self._reconstructSignal(self._sides, self.gap_data)

    def trainSNR(self):
        return tf.reduce_mean(self._pavlovs_SNR(self._target_model.output(), self._reconstructed_input_data,
                                                     onAxis=[1, 2, 3]))

    def _reconstructSignal(self, sides, gaps):
        signal_length = self._window_size - self._gap_length
        first_half = sides[:, :signal_length // 2]
        second_half = sides[:, signal_length // 2:]

        reconstructed_signal = tf.concat([first_half, gaps, second_half], axis=1)
        return reconstructed_signal

    def _loss_graph(self):
        with tf.variable_scope("Loss"):
            gap_stft = self._target_model.output()

            norm_orig = self._squaredEuclideanNorm(gap_stft, onAxis=[1, 2, 3])
            norm_orig_summary = tf.summary.scalar("norm_orig", tf.reduce_min(norm_orig))

            error = gap_stft - self._reconstructed_input_data
            # Nati comment: here you should use only one reduce sum function
            error_per_example = tf.reduce_sum(tf.square(error), axis=[1, 2, 3])

            reconstruction_loss = 0.5 * tf.reduce_sum(error_per_example * (1 + 5 / (norm_orig+1e-2)))

            rec_loss_summary = tf.summary.scalar("reconstruction_loss", reconstruction_loss)

            trainable_vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]) * 1e-2
            l2_loss_summary = tf.summary.scalar("lossL2", lossL2)

            total_loss = tf.add_n([reconstruction_loss, lossL2])
            total_loss_summary = tf.summary.scalar("total_loss", total_loss)

            self._lossSummaries = tf.summary.merge([rec_loss_summary, l2_loss_summary, norm_orig_summary, total_loss_summary])

            return total_loss

    def reconstructAudio(self, audios, model_num=None, max_batchs=200):
        with tf.Session() as sess:
            if model_num is not None:
                path = self.modelsPath(model_num)
            else:
                path = self.modelsPath(self._initial_model_num)
            saver = tf.train.Saver()
            saver.restore(sess, path)
            print("Model restored.")

            batches_count = int(len(audios) / self._batch_size)

            reconstructed = StrechableNumpyArray()
            for batch_num in range(min(batches_count, max_batchs)):
                batch_data = audios[batch_num * self._batch_size:batch_num * self._batch_size + self._batch_size]
                feed_dict = {self._model.input(): batch_data, self._model.isTraining(): False}
                reconstructed_input = sess.run([self._reconstructed_input_data],
                                                         feed_dict=feed_dict)
                reconstructed.append(np.reshape(reconstructed_input, (-1)))
            reconstructed = reconstructed.finalize()
            output_shape = self._target_model.output().shape.as_list()
            output_shape[0] = -1
            reconstructed_stft = np.reshape(reconstructed, output_shape)
            return reconstructed_stft

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
            reconstructed_signal = sess.run(self._reconstructedSignal,
                                            feed_dict={self._sides: sides, self.gap_data: gaps})
            gap_stft = self._target_model.output()

            feed_dict = {self._model.input(): reconstructed_signal, self._target_model.input(): reconstructed_signal,
                         self._model.isTraining(): False}
            reconstructed_input, original = sess.run([self._reconstructed_input_data, gap_stft], feed_dict=feed_dict)
            out_gaps.append(np.reshape(original, (-1)))
            reconstructed.append(np.reshape(reconstructed_input, (-1)))

        output_shape = self._target_model.output().shape.as_list()
        output_shape[0] = -1
        reconstructed = reconstructed.finalize()
        reconstructed = np.reshape(reconstructed, output_shape)
        out_gaps = out_gaps.finalize()
        out_gaps = np.reshape(out_gaps, output_shape)

        data_reader.finish()

        return reconstructed, out_gaps

    def _evaluateValidSNR(self, summaries_dict, validReader, evalWriter, writer, sess, step):
        reconstructed, out_gaps = self._reconstruct(sess, validReader, max_steps=8)
        step_valid_SNR = evalWriter.evaluateImages(reconstructed, out_gaps, self._initial_model_num + step)
        validSNRSummaryToWrite = sess.run(summaries_dict['valid_SNR_summary'],
                                          feed_dict={summaries_dict['valid_SNR']: step_valid_SNR})
        writer.add_summary(validSNRSummaryToWrite, self._initial_model_num + step)

    def _evaluatePlotSummary(self, plot_summary, gaps, feed_dict, writer, sess, step):
        pass

    def _trainingFeedDict(self, sides, gaps, sess):
        rec = sess.run(self._reconstructedSignal, feed_dict={self._sides: sides, self.gap_data: gaps})
        return {self._model.input(): rec, self._target_model.input(): rec, self._model.isTraining(): True}


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None
