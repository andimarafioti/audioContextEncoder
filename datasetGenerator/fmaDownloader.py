import os
import shutil
from datasetGenerator.downloader import Downloader

__author__ = 'Andres'


class FMADownloader(Downloader):
    SMALL_LINK = 'https://os.unil.cloud.switch.ch/fma/fma_small.zip'
    SMALL_FILENAME = 'fma_small.zip'
    SMALL_DIR = SMALL_FILENAME[:-4]
    DIRS = {'train': 'FMA-train', 'valid': 'FMA-valid', 'test': 'FMA-test'}

    def _downloadLinksAndFilenames(self):
        return [(self.SMALL_LINK, self.SMALL_FILENAME)]

    def _extractCompressedFile(self, filename):
        self._extractZip(filename)

    def _divideDataIntoTrainValidAndTestSubsets(self):
        print('Dividing FMA dataset into training, validation and testing subsets.')
        for dirKey in self.DIRS:
            try:
                os.mkdir(self.DIRS[dirKey])
            except FileExistsError as e:
                print('Directory already existed, proceed with caution.\nException:', e)

        i = 0
        for path, directory_name, file_names in os.walk(self.SMALL_DIR):
            for file_name in file_names:
                i += 1
                if i < 7:
                    os.rename(path + '/' + file_name, self.DIRS['train'] + '/' + file_name)
                elif i < 9:
                    os.rename(path + '/' + file_name, self.DIRS['valid'] + '/' + file_name)
                elif i == 9:
                    os.rename(path + '/' + file_name, self.DIRS['test'] + '/' + file_name)
                    i = 0
        shutil.rmtree(self.SMALL_FILENAME[:-4])


if __name__ == "__main__":
    down = FMADownloader()
    down.downloadAndExtract()
