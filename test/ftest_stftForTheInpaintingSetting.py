from unittest import TestCase

import numpy as np
import tensorflow as tf

from system.preAndPostProcessor import PreAndPostProcessor

__author__ = 'Andres'


class TestStftForTheContextEncoder(TestCase):
    def setUp(self):
        self.signal_length = 5120
        self.gap_length = 1024
        self.fft_window_length = 512
        self.fft_hop_size = 128

        self.anStftForTheInpaintingSetting = PreAndPostProcessor(signalLength=self.signal_length,
                                                                 gapLength=self.gap_length,
                                                                 fftWindowLength=self.fft_window_length,
                                                                 fftHopSize=self.fft_hop_size)

    def test01TheStftTakesTheInpaintingParametersAsInput(self):
        self.assertEquals(self.anStftForTheInpaintingSetting.signalLength(), self.signal_length)
        self.assertEquals(self.anStftForTheInpaintingSetting.gapLength(), self.gap_length)
        self.assertEquals(self.anStftForTheInpaintingSetting.fftWindowLenght(), self.fft_window_length)
        self.assertEquals(self.anStftForTheInpaintingSetting.fftHopSize(), self.fft_hop_size)

    def test02TheStftKnowsHowMuchPaddingItShouldApply(self):
        self.assertEquals(self.anStftForTheInpaintingSetting.padding(), self.fft_window_length-self.fft_hop_size)

        fft_window_length = 1024
        fft_hop_size = 128
        anStftForTheInpaintingSetting = PreAndPostProcessor(signalLength=self.signal_length,
                                                            gapLength=self.gap_length,
                                                            fftWindowLength=fft_window_length,
                                                            fftHopSize=fft_hop_size)
        self.assertEquals(anStftForTheInpaintingSetting.padding(), fft_window_length - fft_hop_size)

        fft_window_length = 1024
        fft_hop_size = 256
        anStftForTheInpaintingSetting = PreAndPostProcessor(signalLength=self.signal_length,
                                                            gapLength=self.gap_length,
                                                            fftWindowLength=fft_window_length,
                                                            fftHopSize=fft_hop_size)
        self.assertEquals(anStftForTheInpaintingSetting.padding(), fft_window_length - fft_hop_size)

    def test03TheStftKnowsWhatSignalItShouldTakeForTheSTFTOfTheGap(self):
        fake_batch_of_signal = np.array([np.arange(self.signal_length)])
        produced_signal = self.anStftForTheInpaintingSetting._removeExtraSidesForSTFTOfGap(fake_batch_of_signal)

        gap_begins = (self.signal_length-self.gap_length)//2
        gap_ends = gap_begins + self.gap_length
        padding = self.fft_window_length-self.fft_hop_size

        np.testing.assert_almost_equal(fake_batch_of_signal[:, gap_begins - padding:gap_ends + padding], produced_signal)

        fft_window_length = 128
        fft_hop_size = 32

        anStftForTheInpaintingSetting = PreAndPostProcessor(signalLength=self.signal_length,
                                                            gapLength=self.gap_length,
                                                            fftWindowLength=fft_window_length,
                                                            fftHopSize=fft_hop_size)
        produced_signal = anStftForTheInpaintingSetting._removeExtraSidesForSTFTOfGap(fake_batch_of_signal)
        padding = fft_window_length - fft_hop_size
        np.testing.assert_almost_equal(fake_batch_of_signal[:, gap_begins - padding:gap_ends + padding], produced_signal)

    def test04TheStftProducesAnSTFTOfTheExpectedShapeForTheGap(self):
        batch_size = 32
        aBatchOfSignals = tf.placeholder(tf.float32, shape=(batch_size, self.signal_length), name='input_data')
        aStft = self.anStftForTheInpaintingSetting.stftForGapOf(aBatchOfSignals)

        framesOnGap = (((self.gap_length + self.anStftForTheInpaintingSetting.padding()*2)-self.fft_window_length)/
                       self.fft_hop_size)+1
        binsPerFrame = self.fft_window_length//2+1
        realAndImagChannels = 2
        self.assertEquals(aStft.shape.as_list(), [32, framesOnGap, binsPerFrame, realAndImagChannels])

    def test05TheStftRemovesTheGapCorrectly(self):
        fake_batch_of_signal = np.array([np.arange(self.signal_length)])
        produced_signal = self.anStftForTheInpaintingSetting._removeGap(fake_batch_of_signal)

        gap_begins = (self.signal_length-self.gap_length)//2
        gap_ends = gap_begins + self.gap_length

        left_side = fake_batch_of_signal[:, :gap_begins]
        right_side = fake_batch_of_signal[:, gap_ends:]
        signal_without_gap = tf.stack((left_side, right_side), axis=1)

        with tf.Session() as sess:
            produced_signal, signal_without_gap = sess.run([produced_signal, signal_without_gap])

        np.testing.assert_almost_equal(signal_without_gap, produced_signal)

    def test06TheStftAddsTheCorrectPaddingToTheSides(self):
        side_length = (self.signal_length-self.gap_length)//2

        left_side = np.array([np.arange(side_length, dtype=np.float32)])
        right_side = np.array([np.arange(side_length, dtype=np.float32)])
        fake_batch_of_sides = tf.stack((left_side, right_side), axis=1)

        produced_signal = self.anStftForTheInpaintingSetting._addPaddingForStftOfContext(fake_batch_of_sides)

        with tf.Session() as sess:
            produced_signal = sess.run(produced_signal)

        left_side_padded = np.concatenate((left_side, np.zeros((1, self.fft_window_length-self.fft_hop_size))), axis=1)
        right_side_padded = np.concatenate((right_side, np.zeros((1, self.fft_window_length-self.fft_hop_size))), axis=1)
        new_signal = np.stack([left_side_padded, right_side_padded], axis=1)

        np.testing.assert_almost_equal(new_signal, produced_signal)

    def test07TheStftOfTheContextHasTheExpectedShape(self):
        batch_size = 32
        aBatchOfSignals = tf.placeholder(tf.float32, shape=(batch_size, self.signal_length), name='input_data')
        aStft = self.anStftForTheInpaintingSetting.stftForTheContextOf(aBatchOfSignals)

        side_length = (self.signal_length-self.gap_length)//2
        framesOnSides = ((side_length + self.anStftForTheInpaintingSetting.padding() - self.fft_window_length)
                         / self.fft_hop_size)+1
        binsPerFrame = self.fft_window_length//2+1
        realAndImagChannels = 2
        beforeAndAfterChannels = 2

        self.assertEquals(aStft.shape.as_list(), [32, framesOnSides, binsPerFrame,
                                                  realAndImagChannels*beforeAndAfterChannels])

    def test08TheStftProducesTheCorrectShapeWhenDoingTheInverseStftOnTheGap(self):
        batch_size = 32
        framesOnGap = (((self.gap_length + self.anStftForTheInpaintingSetting.padding()*2)-self.fft_window_length)/
                       self.fft_hop_size)+1
        binsPerFrame = self.fft_window_length//2+1
        batchOfGapStft = tf.zeros((batch_size, framesOnGap, binsPerFrame), dtype=tf.complex64)

        batchOfGaps = self.anStftForTheInpaintingSetting.inverseStftOfGap(batchOfGapStft)

        with tf.Session() as sess:
            batchOfGaps = sess.run(batchOfGaps)

        self.assertEquals(batchOfGaps.shape, (batch_size, self.gap_length))

    def test09TheStftProducesTheCorrectShapeWhenDoingTheInverseStftOnTheFullSignal(self):
        batch_size = 32
        frameCount = ((self.signal_length-self.fft_window_length)/self.fft_hop_size)+1
        binsPerFrame = self.fft_window_length//2+1
        batchOfSignalStft = tf.zeros((batch_size, frameCount, binsPerFrame), dtype=tf.complex64)

        batchOfSignals = self.anStftForTheInpaintingSetting.inverseStftOfSignal(batchOfSignalStft)

        with tf.Session() as sess:
            batchOfGaps = sess.run(batchOfSignals)

        self.assertEquals(batchOfGaps.shape, (batch_size, self.signal_length))
