import tensorflow as tf
import numpy as np
from system.dnnSystem import DNNSystem
from utils.colorize import colorize
from utils.strechableNumpyArray import StrechableNumpyArray
from utils.tfReader import TFReader

__author__ = 'Andres'


class ContextEncoderSystem(DNNSystem):
    def __init__(self, architecture, batchSize, aPreProcessor, name):
        self._windowSize = aPreProcessor.signalLength()
        self._batchSize = batchSize
        self._audio = tf.placeholder(tf.float32, shape=(batchSize, self._windowSize), name='audio_data')
        self._preProcessForGap = aPreProcessor.stftForGapOf(self._audio)
        self._preProcessForContext = aPreProcessor.stftForTheContextOf(self._audio)
        super().__init__(architecture, name)
        self._SNR = tf.reduce_mean(self._pavlovs_SNR(self._architecture.output(), self._architecture.target()))

    def optimizer(self, learningRate):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer(learning_rate=learningRate).minimize(self._architecture.loss())

    def _feedDict(self, data, sess, isTraining=True):
        net_input, net_target = sess.run([self._preProcessForContext, self._preProcessForGap], feed_dict={self._audio: data})
        return {self._architecture.input(): net_input, self._architecture.target(): net_target,
                self._architecture.isTraining(): isTraining}

    def reconstruct(self, data_path, model_num, max_steps=200):
        with tf.Session() as sess:
            reader = self._loadReader(data_path)
            path = self.modelsPath(model_num)
            saver = tf.train.Saver()
            saver.restore(sess, path)
            print("Model restored.")
            sess.run([tf.local_variables_initializer()])
            return self._reconstruct(sess, reader, max_steps)

    def _reconstruct(self, sess, data_reader, max_steps):
        data_reader.start()
        reconstructed = StrechableNumpyArray()
        out_gaps = StrechableNumpyArray()
        input_shape = self._architecture.inputShape()
        contexts = np.empty(input_shape)

        for batch_num in range(max_steps):
            try:
                audio = data_reader.dataOperation(session=sess)
            except StopIteration:
                print("rec End of queue!", batch_num)
                break

            feed_dict = self._feedDict(audio, sess, False)
            reconstructed_input, original, context = sess.run([self._architecture.output(), self._architecture.target(),
                                                      self._architecture.input()],
                                                     feed_dict=feed_dict)
            out_gaps.append(np.reshape(original, (-1)))
            reconstructed.append(np.reshape(reconstructed_input, (-1)))
            contexts = np.concatenate([contexts, context], axis=0)

        output_shape = self._architecture.output().shape.as_list()
        output_shape[0] = -1
        reconstructed = reconstructed.finalize()
        reconstructed = np.reshape(reconstructed, output_shape)
        out_gaps = out_gaps.finalize()
        out_gaps = np.reshape(out_gaps, output_shape)

        data_reader.finish()

        return reconstructed, out_gaps, contexts

    def _evaluate(self, summariesDict, feed_dict, validReader, sess):
        trainSNRSummaryToWrite = sess.run(summariesDict['train_SNR_summary'], feed_dict=feed_dict)

        try:
            audio = validReader.dataOperation(session=sess)
        except StopIteration:
            print("valid End of queue!")
            return [trainSNRSummaryToWrite]
        feed_dict = self._feedDict(audio, sess, False)
        validSNRSummary = sess.run(summariesDict['valid_SNR_summary'], feed_dict)
        iamgeSummary = sess.run(summariesDict['image_summaries'], feed_dict)

        return [trainSNRSummaryToWrite, validSNRSummary, iamgeSummary]

    def _loadReader(self, dataPath):
        return TFReader(dataPath, self._windowSize, batchSize=self._batchSize, capacity=int(2e5), num_epochs=400)

    def _evaluationSummaries(self):
        summaries_dict = {'train_SNR_summary': tf.summary.scalar("training_SNR", self._SNR),
                          'valid_SNR_summary': tf.summary.scalar("validation_SNR", self._SNR),
    			          'image_summaries': self._spectrogramImageSummary()}
        return summaries_dict

    def _squaredEuclideanNorm(self, tensor, onAxis=[1, 2, 3]):
        squared = tf.square(tensor)
        summed = tf.reduce_sum(squared, axis=onAxis)
        return summed

    def _log10(self, tensor):
        numerator = tf.log(tensor)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def _pavlovs_SNR(self, y_orig, y_inp, onAxis=[1, 2, 3]):
        norm_y_orig = self._squaredEuclideanNorm(y_orig, onAxis)
        norm_y_orig_minus_y_inp = self._squaredEuclideanNorm(y_orig - y_inp, onAxis)
        return 10 * self._log10(norm_y_orig / norm_y_orig_minus_y_inp)

    def _spectrogramImageSummary(self):
        complexOutput = self._architecture.output()[0]
        outputSpectrogram = tf.sqrt(tf.square(complexOutput[:, :, 0]) + tf.square(complexOutput[:, :, 1]))

        complexTarget = self._architecture.target()[0]
        targetSpectrogram = tf.sqrt(tf.square(complexTarget[:, :, 0]) + tf.square(complexTarget[:, :, 1]))

        complexLeft = self._architecture.input()[0, :, :, 0:2]
        leftSpectrogram = tf.sqrt(tf.square(complexLeft[:, :, 0]) + tf.square(complexLeft[:, :, 1]))

        complexRight = self._architecture.input()[0, :, :, 2:4]
        rightSpectrogram = tf.sqrt(tf.square(complexRight[:, :, 0]) + tf.square(complexRight[:, :, 1]))

        totalSpectrogram = tf.transpose(tf.concat([leftSpectrogram, outputSpectrogram,
                                                   rightSpectrogram], axis=0))

        return tf.summary.merge([tf.summary.image("Original", [colorize(tf.transpose(targetSpectrogram))]),
                                tf.summary.image("Generated", [colorize(tf.transpose(outputSpectrogram))]),
                                tf.summary.image("Complete", [colorize(totalSpectrogram)])])
