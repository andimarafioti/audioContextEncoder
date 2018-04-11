from datasetGenerator.downloader import Downloader

__author__ = 'Andres'


class NSynthDownloader(Downloader):
    TRAIN_LINK = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz"
    VALID_LINK = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz"
    TEST_LINK = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"

    TRAIN_FILENAME = "nsynth_train.tar.gz"
    VALID_FILENAME = "nsynth_valid.tar.gz"
    TEST_FILENAME = "nsynth_test.tar.gz"

    def downloadAndExtract(self):
        self._download(self.TEST_LINK, self.TEST_FILENAME)
        self._extractTar(self.TEST_FILENAME)
        self._deleteCompressedFile(self.TEST_FILENAME)
        self._download(self.TRAIN_LINK, self.TRAIN_FILENAME)
        self._extractTar(self.TRAIN_FILENAME)
        self._deleteCompressedFile(self.TRAIN_FILENAME)
        self._download(self.VALID_LINK, self.VALID_FILENAME)
        self._extractTar(self.VALID_FILENAME)
        self._deleteCompressedFile(self.VALID_FILENAME)
