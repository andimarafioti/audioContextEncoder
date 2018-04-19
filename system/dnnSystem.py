import tensorflow as tf
import re

__author__ = 'Andres'


class DNNSystem(object):
    def __init__(self, architecture, name):
        self._architecture = architecture
        self._name = name

    def optimizer(self, learningRate):
        raise NotImplementedError("Subclass Responsibility")

    def _feedDict(self, data, sess, isTraining=True):
        raise NotImplementedError("Subclass Responsibility")

    def _evaluate(self, summariesDict, feed_dict, validReader, sess):
        raise NotImplementedError("Subclass Responsibility")

    def _loadReader(self, dataPath):
        raise NotImplementedError("Subclass Responsibility")

    def _evaluationSummaries(self):
        raise NotImplementedError("Subclass Responsibility")

    def train(self, trainTFRecordPath, validTFRecordPath, learningRate, numSteps=6e5, restoreNum=None):
        with tf.Session() as sess:
            trainReader = self._loadReader(trainTFRecordPath)
            validReader = self._loadReader(validTFRecordPath)
            optimizer = self.optimizer(learningRate)

            saver = tf.train.Saver(max_to_keep=100)
            path = self.modelsPath(restoreNum)
            _modelNum = get_trailing_number(path[:-5])

            if _modelNum == 0:
                init = tf.global_variables_initializer()
                sess.run([init, tf.local_variables_initializer()])
                print("Initialized")
            else:
                saver.restore(sess, path)
                sess.run([tf.local_variables_initializer()])
                print("Model restored.")

            logs_path = 'utils/logdir/' + self._name  # write each run to a diff folder.
            print("logs path:", logs_path)
            writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

            summariesDict = self._evaluationSummaries()

            try:
                trainReader.start()
                validReader.start()

                for step in range(1, int(numSteps)):
                    try:
                        data = trainReader.dataOperation(session=sess)
                    except StopIteration:
                        print("End of queue at step", step)
                        break

                    feed_dict = self._feedDict(data, sess, isTraining=True)
                    sess.run(optimizer, feed_dict=feed_dict)

                    if step % 40 == 0:
                        train_summ = sess.run(self._architecture.lossSummaries(), feed_dict=feed_dict)
                        writer.add_summary(train_summ, _modelNum + step)
                    if step % 2000 == 0:
                        summaries = self._evaluate(summariesDict, feed_dict, validReader, sess)
                        for summary in summaries:
                            writer.add_summary(summary, _modelNum+step)
                        saver.save(sess, self.modelsPath(_modelNum + step))
            except KeyboardInterrupt:
                pass

            saver.save(sess, self.modelsPath(_modelNum + step))
            trainReader.finish()
            validReader.finish()
            print("Finalizing at step:", _modelNum + step)
            print("Last saved model:", self.modelsPath(_modelNum + step))

    def modelsPath(self, models_number=None):
        pathdir = "utils/saved_models/" + self._name
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


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None
