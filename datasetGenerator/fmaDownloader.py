import os
import shutil
from datasetGenerator.downloader import Downloader

__author__ = 'Andres'


class FMADownloader(Downloader):
    SMALL_LINK = 'https://os.unil.cloud.switch.ch/fma/fma_small.zip'
    SMALL_FILENAME = 'fma_small.zip'
    SMALL_DIR = SMALL_FILENAME[:-4]

    TRAIN_DIR = 'FMA-train'
    VALID_DIR = 'FMA-valid'
    TEST_DIR = 'FMA-test'

    def _downloadLinksAndFilenames(self):
        return [(self.SMALL_LINK, self.SMALL_FILENAME)]

    def _extractCompressedFile(self, filename):
        self._extractZip(filename)

    def _divideDataIntoTrainValidAndTestSubsets(self):
        print('Dividing FMA dataset into training, validation and testing subsets.')
        for dir_name in [self.TRAIN_DIR, self.VALID_DIR, self.TEST_DIR]:
            try:
                os.mkdir(dir_name)
            except FileExistsError as e:
                print('Directory already existed, proceed with caution.\nException:', e)

        i = 0
        for path, directory_name, file_names in os.walk(self.SMALL_DIR):
            for file_name in file_names:
                i += 1
                if i < 7:
                    os.rename(path + '/' + file_name, self.TRAIN_DIR + '/' + file_name)
                elif i < 9:
                    os.rename(path + '/' + file_name, self.VALID_DIR + '/' + file_name)
                elif i == 9:
                    os.rename(path + '/' + file_name, self.TEST_DIR + '/' + file_name)
                    i = 0
        shutil.rmtree(self.SMALL_DIR)


if __name__ == "__main__":
    down = FMADownloader()
    down.downloadAndExtract()
