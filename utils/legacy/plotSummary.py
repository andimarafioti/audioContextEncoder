import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import tensorflow as tf

__author__ = 'Andres'


class PlotSummary(object):
    def __init__(self, name):
        self._name = name
        self._placeholder = tf.placeholder(tf.uint8, (None, None, None, None))
        self._summary = tf.summary.image(name, self._placeholder)
        self._image = None

    def produceSummaryToWrite(self, session):
        decoded_image = session.run(self._image)
        feed_dict = {self._placeholder: decoded_image}
        return session.run(self._summary, feed_dict=feed_dict)

    def plotSideBySide(self, out_gaps, reconstructed):
        f, axarr = plt.subplots(4, 2, sharey='row')
        f.set_size_inches(14, 24)
        stop_value = 4
        for i in range(0, stop_value):
            axarr[i, 0].plot(out_gaps[i])
            axarr[i, 1].plot(reconstructed[i])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        self._image = image