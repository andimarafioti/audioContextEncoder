import tensorflow as tf
import numpy as np
from evaluationWriter import EvaluationWriter
from strechableNumpyArray import StrechableNumpyArray
from tfReader import TFReader


class ConditionalAutoEncoderNetwork(object):
    def __init__(self, batch_size, window_size, gap_length, name, latent_space_dimentionality=64):
        self._batch_size = batch_size
        self._window_size = window_size
        self._gap_length = gap_length
        self._name = name
        self._latent_dimentionality = latent_space_dimentionality
        self._initial_model_num = 0
        # tf Graph input
        self.train_input_data = tf.placeholder(tf.float32, shape=(batch_size, window_size - gap_length),
                                               name='train_input_data')
        self.gap_data = tf.placeholder(tf.float32, shape=(batch_size, gap_length), name='gap_data')

        self._reconstructed_input_data = self._network(self.train_input_data, isTraining=True)

        # Loss, Optimizer and Predictions

        self._loss = self._loss_graph()
        self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self._loss)

    def _network(self, dataset, isTraining):
        encoded = self._encoder(dataset, isTraining)
        reconstructed = self._decoder(encoded, isTraining)
        return reconstructed

    def _encoder(self, data, isTraining):
        with tf.variable_scope("Encoder"):
            data = data - 0.5
            reshape = tf.reshape(data, (self._batch_size, self._window_size - self._gap_length, 1))

            first_conv = self._convLayer(input_signal=reshape, filter_width=129, input_channels=1,
                                         output_channels=16, stride=4, name="First_Conv", isTraining=isTraining)
            second_conv = self._convLayer(input_signal=first_conv, filter_width=65, input_channels=16,
                                          output_channels=64, stride=4, name="Second_Conv", isTraining=isTraining)
            third_conv = self._convLayer(input_signal=second_conv, filter_width=33, input_channels=64,
                                         output_channels=256, stride=4, name="Third_Conv", isTraining=isTraining)
            fourth_conv = self._convLayer(input_signal=third_conv, filter_width=17, input_channels=256,
                                         output_channels=1024, stride=4, name="Fourth_Conv", isTraining=isTraining)
            last_conv = self._convLayer(input_signal=fourth_conv, filter_width=9, input_channels=1024,
                                          output_channels=4096, stride=4, name="Last_Conv", isTraining=isTraining)

            return last_conv

    def _decoder(self, data, isTraining):
        with tf.variable_scope("Decoder"):
            with tf.variable_scope('Decoding', reuse=not isTraining):
                layers_filters = self._weight_variable([5, 4096, 1024])
                layers_biases = self._bias_variable([1024])
                conv = tf.nn.conv1d(data, layers_filters, stride=4, padding="SAME") + layers_biases
                shape = conv.get_shape().as_list()
                print("decoded ", shape)
                print(self._gap_length)
                reshape = tf.reshape(conv, [shape[0], self._gap_length])
                reshape = reshape + 0.5
            return reshape

    def _convLayer(self, input_signal, filter_width, input_channels, output_channels, stride, name, isTraining,
                   padding="SAME"):
        with tf.variable_scope(name, reuse=not isTraining):
            layers_filters = self._weight_variable([filter_width, input_channels, output_channels])
            layers_biases = self._bias_variable([output_channels])
            conv = tf.nn.conv1d(input_signal, layers_filters, stride=stride, padding=padding)
            return tf.nn.relu(conv + layers_biases)

    def _linearLayer(self, input_signal, input_size, output_size, name, isTraining):
        with tf.variable_scope(name, reuse=not isTraining):
            weights = self._weight_variable([input_size, output_size])
            biases = self._bias_variable(output_size)
            linear_function = tf.matmul(input_signal, weights) + biases
            return linear_function

    def _weight_variable(self, shape):
        return tf.get_variable('W', shape, initializer=tf.contrib.layers.xavier_initializer())

    def _bias_variable(self, shape):
        return tf.get_variable('bias', shape, initializer=tf.contrib.layers.xavier_initializer())

    def euclideanNorm(self, tensor):
        squared = tf.square(tensor)
        summed = tf.reduce_sum(squared, axis=1)
        return tf.sqrt(summed + 1e-10)

    def _loss_graph(self):
        with tf.variable_scope("Loss"):
            norm_orig = self.euclideanNorm((self.gap_data - 0.5) * 2)
            error = (self.gap_data - self._reconstructed_input_data) * 2
            reconstruction_loss = 0.5 * tf.reduce_sum(tf.reduce_sum(tf.square(error), axis=1) * (1 + 1 / norm_orig))
            tf.summary.scalar("reconstruction_loss", reconstruction_loss)

            trainable_vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]) * 1e1
            tf.summary.scalar("lossL2", lossL2)

            total_loss = tf.add_n([reconstruction_loss, lossL2])
            tf.summary.scalar("total_loss", total_loss)

            return reconstruction_loss

    def modelsPath(self, models_number):
        models_path = "saved_models/model-" + self._name
        models_ext = ".ckpt"
        return models_path + str(models_number) + models_ext

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

            feed_dict = {self.train_input_data: sides, self.gap_data: gaps}
            reconstructed.append(np.reshape(sess.run(self._reconstructed_input_data, feed_dict=feed_dict), (-1)))
        reconstructed = reconstructed.finalize()
        reconstructed = np.reshape(reconstructed, (-1, self._gap_length))

        out_gaps = out_gaps.finalize()
        out_gaps = np.reshape(out_gaps, (-1, self._gap_length))
        data_reader.finish()

        return reconstructed, out_gaps

    def train(self, train_data_path, valid_data_path, num_steps=2e2, restore_num=None):
        with tf.Session() as sess:
            try:
                trainReader = TFReader(train_data_path, self._window_size, self._gap_length, capacity=int(1e6), num_epochs=40)
                validReader = TFReader(valid_data_path, self._window_size, self._gap_length, capacity=int(1e6), num_epochs=4000)

                saver = tf.train.Saver(max_to_keep=1000)
                if restore_num:
                    path = self.modelsPath(restore_num)
                    self._initial_model_num = restore_num
                    saver.restore(sess, path)
                    sess.run([tf.local_variables_initializer()])
                    print("Model restored.")
                else:
                    init = tf.global_variables_initializer()
                    sess.run([init, tf.local_variables_initializer()])
                    print("Initialized")

                logs_path = 'logdir_real_cae/' + self._name  # write each run to a diff folder.
                print("logs path:", logs_path)
                writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
                merged_summary = tf.summary.merge_all()

                trainReader.start()
                evalWriter = EvaluationWriter(self._name + '.xlsx')

                for step in range(1, int(num_steps)):
                    try:
                        sides, gaps = trainReader.dataOperation(session=sess)
                    except StopIteration:
                        print(step)
                        print("End of queue!")
                        break

                    feed_dict = {self.train_input_data: sides, self.gap_data: gaps}
                    sess.run(self._optimizer, feed_dict=feed_dict)

                    if step % 40 == 0:
                        train_summ = sess.run(merged_summary, feed_dict=feed_dict)
                        writer.add_summary(train_summ, self._initial_model_num + step)
                    if step % 2000 == 0:
                        saver.save(sess, self.modelsPath(self._initial_model_num + step))
                        reconstructed, out_gaps = self._reconstruct(sess, validReader, max_steps=256)
                        evalWriter.evaluate(reconstructed, out_gaps, self._initial_model_num + step)

            except KeyboardInterrupt:
                pass
            evalWriter.save()
            train_summ = sess.run([merged_summary], feed_dict=feed_dict)[0]
            writer.add_summary(train_summ, self._initial_model_num + step)
            saver.save(sess, self.modelsPath(self._initial_model_num + step))
            self._initial_model_num += step

            trainReader.finish()
            print("Finalizing at step:", self._initial_model_num)
            print("Last saved model:", self.modelsPath(self._initial_model_num))
			
tf.reset_default_graph()

train_filename = 'train_full_w5120_g1024_h512_19404621.tfrecords'
valid_filename = 'valid_full_w5120_g1024_h512_ex913967.tfrecords'

anAutoEncoderNetwork = ConditionalAutoEncoderNetwork(batch_size=256, window_size=5120, gap_length=1024,
                                                     name='ninth', latent_space_dimentionality=32000)
anAutoEncoderNetwork.train(train_filename, valid_filename, num_steps=100001, restore_num=464000)
