import re

import numpy as np
import tensorflow as tf

from utils.legacy.evaluationWriter import EvaluationWriter
from utils.legacy.plotSummary import PlotSummary
from utils.strechableNumpyArray import StrechableNumpyArray
from utils.tfReader import TFReader

__author__ = 'Andres'


class ContextEncoderNetwork(object):
    def __init__(self, model, batch_size, window_size, gap_length, learning_rate, name):
        self._batch_size = batch_size
        self._window_size = window_size
        self._gap_length = gap_length
        self._name = name
        self._initial_model_num = 0

        self._model = model
        self.gap_data = tf.placeholder(tf.float32, shape=(batch_size, gap_length), name='gap_data')

        self._reconstructed_input_data = self._model.output()

        self._loss = self._loss_graph()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._loss)
        self._SNR = self.trainSNR()

    def trainSNR(self):
        return tf.reduce_mean(self._pavlovs_SNR(self.gap_data, self._reconstructed_input_data))

    def _squaredEuclideanNorm(self, tensor, onAxis=[1]):
        squared = tf.square(tensor)
        summed = tf.reduce_sum(squared, axis=onAxis)
        return summed

    def _log10(self, tensor):
        numerator = tf.log(tensor)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def _pavlovs_SNR(self, y_orig, y_inp, onAxis=[1]):
        norm_y_orig = self._squaredEuclideanNorm(y_orig, onAxis)
        norm_y_orig_minus_y_inp = self._squaredEuclideanNorm(y_orig - y_inp, onAxis)
        return 10 * self._log10(norm_y_orig / norm_y_orig_minus_y_inp)

    def _loss_graph(self):
        with tf.variable_scope("Loss"):
            norm_orig = self._squaredEuclideanNorm(self.gap_data) / 5
            error = self.gap_data - self._reconstructed_input_data
            reconstruction_loss = 0.5 * tf.reduce_sum(tf.reduce_sum(tf.square(error), axis=1) * (1 + 1 / norm_orig))
            rec_loss_summary = tf.summary.scalar("reconstruction_loss", reconstruction_loss)

            trainable_vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]) * 1e-2
            l2_loss_summary = tf.summary.scalar("lossL2", lossL2)

            total_loss = tf.add_n([reconstruction_loss, lossL2])
            total_loss_summary = tf.summary.scalar("total_loss", total_loss)

            self._lossSummaries = tf.summary.merge([rec_loss_summary, l2_loss_summary, total_loss_summary])

            return total_loss

    def modelsPath(self, models_number=None):
        pathdir = "../saved_models/" + self._name
        if models_number is None:
            ckpt = tf.train.get_checkpoint_state(pathdir)
            print(ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                return ckpt.model_checkpoint_path
            else:
                models_number = 0
        models_path = pathdir + "/model-" + self._name
        models_ext = ".ckpt"
        return models_path + str(models_number) + models_ext

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
                reconstructed.append(np.reshape(sess.run(self._reconstructed_input_data, feed_dict=feed_dict), (-1)))
            reconstructed = reconstructed.finalize()
            reconstructed = np.reshape(reconstructed, (-1, self._gap_length))
            return reconstructed

    def reconstruct(self, data_path, model_num=None, max_steps=200):
        with tf.Session() as sess:
            reader = TFReader(data_path, self._window_size, self._gap_length, capacity=int(1e6))
            if model_num is not None:
                path = self.modelsPath(model_num)
            else:
                path = self.modelsPath(self._initial_model_num)
            saver = tf.train.Saver()
            saver.restore(sess, path)
            print("Model restored.")
            sess.run([tf.local_variables_initializer()])
            reconstructed, out_gaps = self._reconstruct(sess, reader, max_steps)
            return reconstructed, out_gaps

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
            out_gaps.append(np.reshape(gaps, (-1)))

            feed_dict = {self._model.input(): sides, self._model.isTraining(): False}
            reconstructed.append(np.reshape(sess.run(self._reconstructed_input_data, feed_dict=feed_dict), (-1)))
        reconstructed = reconstructed.finalize()
        reconstructed = np.reshape(reconstructed, (-1, self._gap_length))

        out_gaps = out_gaps.finalize()
        out_gaps = np.reshape(out_gaps, (-1, self._gap_length))
        data_reader.finish()

        return reconstructed, out_gaps

    def _initEvaluationSummaries(self):
        summaries_dict = {'train_SNR_summary': tf.summary.scalar("training_SNR", self._SNR),
                          'valid_SNR': tf.placeholder(tf.float32, name="valid_SNR"),
                          'plot_summary': PlotSummary('reconstruction')}
        summaries_dict['valid_SNR_summary'] = tf.summary.scalar("validation_SNR", summaries_dict['valid_SNR'])
        return summaries_dict

    def _evaluateTrainingSNR(self, train_SNR_summary, feed_dict, writer, sess, step):
        trainSNRSummaryToWrite = sess.run(train_SNR_summary, feed_dict=feed_dict)
        writer.add_summary(trainSNRSummaryToWrite, self._initial_model_num + step)

    def _evaluateValidSNR(self, summaries_dict, validReader, evalWriter, writer, sess, step):
        reconstructed, out_gaps = self._reconstruct(sess, validReader, max_steps=256)
        step_valid_SNR = evalWriter.evaluate(reconstructed, out_gaps, self._initial_model_num + step)
        validSNRSummaryToWrite = sess.run(summaries_dict['valid_SNR_summary'],
                                          feed_dict={summaries_dict['valid_SNR']: step_valid_SNR})
        writer.add_summary(validSNRSummaryToWrite, self._initial_model_num + step)

    def _evaluatePlotSummary(self, plot_summary, gaps, feed_dict, writer, sess, step):
        reconstructed = sess.run(self._reconstructed_input_data, feed_dict=feed_dict)
        plot_summary.plotSideBySide(gaps, reconstructed)
        summaryToWrite = plot_summary.produceSummaryToWrite(sess)
        writer.add_summary(summaryToWrite, self._initial_model_num + step)

    def _trainingFeedDict(self, sides, gaps, sess):
        return {self._model.input(): sides, self.gap_data: gaps, self._model.isTraining(): True}

    def train(self, train_data_path, valid_data_path, num_steps=2e2, restore_num=None, per_process_gpu_memory_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            try:
                trainReader = TFReader(train_data_path, self._window_size, self._gap_length, capacity=int(2e5), num_epochs=400)
                validReader = TFReader(valid_data_path, self._window_size, self._gap_length, capacity=int(2e5), num_epochs=40000)

                saver = tf.train.Saver(max_to_keep=1000)
                print(restore_num)
                path = self.modelsPath(restore_num)
                self._initial_model_num = get_trailing_number(path[:-5])
                if self._initial_model_num == 0:
                    init = tf.global_variables_initializer()
                    sess.run([init, tf.local_variables_initializer()])
                    print("Initialized")
                else:
                    saver.restore(sess, path)
                    sess.run([tf.local_variables_initializer()])
                    print("Model restored.")

                logs_path = '../logdir_real_cae/' + self._name  # write each run to a diff folder.
                print("logs path:", logs_path)
                writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

                summaries_dict = self._initEvaluationSummaries()

                trainReader.start()
                evalWriter = EvaluationWriter(self._name + '.xlsx')

                # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                # many_runs_timeline = TimeLiner()

                for step in range(1, int(num_steps)):
                    try:
                        sides, gaps = trainReader.dataOperation(session=sess)
                    except StopIteration:
                        print(step)
                        print("End of queue!")
                        break

                    feed_dict = self._trainingFeedDict(sides, gaps, sess)
                    sess.run(self._optimizer, feed_dict=feed_dict)  # , options=options, run_metadata=run_metadata)

                    # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    # many_runs_timeline.update_timeline(chrome_trace)

                    if step % 40 == 0:
                        train_summ = sess.run(self._lossSummaries, feed_dict=feed_dict)
                        print("Training summaries: {}".format(train_summ))
                        writer.add_summary(train_summ, self._initial_model_num + step)
                    if step % 2000 == 0:
                        print(step)
                        self._evaluateTrainingSNR(summaries_dict['train_SNR_summary'], feed_dict, writer, sess, step)
                        self._evaluatePlotSummary(summaries_dict['plot_summary'], gaps, feed_dict, writer, sess, step)
                        self._evaluateValidSNR(summaries_dict, validReader, evalWriter, writer, sess, step)
                        saver.save(sess, self.modelsPath(self._initial_model_num + step))

            except KeyboardInterrupt:
                pass
            # many_runs_timeline.save('timeline_03_merged_%d_runs.json' % step)
            evalWriter.save()
            train_summ = sess.run(self._lossSummaries, feed_dict=feed_dict)
            writer.add_summary(train_summ, self._initial_model_num + step)
            saver.save(sess, self.modelsPath(self._initial_model_num + step))
            self._initial_model_num += step

            trainReader.finish()
            print("Finalizing at step:", self._initial_model_num)
            print("Last saved model:", self.modelsPath(self._initial_model_num))



def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None
