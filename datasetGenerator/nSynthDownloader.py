from datasetGenerator.downloader import Downloader

__author__ = 'Andres'


class NSynthDownloader(Downloader):
    TRAIN_LINK = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz"
    VALID_LINK = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz"
    TEST_LINK = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"

    TRAIN_FILENAME = "nsynth_train.tar.gz"
    VALID_FILENAME = "nsynth_valid.tar.gz"
    TEST_FILENAME = "nsynth_test.tar.gz"

    TRAIN_DIR = "nsynth-train/audio"
    VALID_DIR = "nsynth-valid/audio"
    TEST_DIR = "nsynth-test/audio"

    def _downloadLinksAndFilenames(self):
        return [(self.TEST_LINK, self.TEST_FILENAME),
                (self.TRAIN_LINK, self.TRAIN_FILENAME),
                (self.VALID_LINK, self.VALID_FILENAME)]

    def _extractCompressedFile(self, filename):
        self._extractTar(filename)

    def _divideDataIntoTrainValidAndTestSubsets(self):
        print('NSynth dataset comes divided into training, validation and testing subsets.')

if __name__ == "__main__":
    down = NSynthDownloader()
    down.downloadAndExtract()
