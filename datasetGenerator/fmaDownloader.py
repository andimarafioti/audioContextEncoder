from datasetGenerator.downloader import Downloader

__author__ = 'Andres'


class FMADownloader(Downloader):
    SMALL_LINK = 'https://os.unil.cloud.switch.ch/fma/fma_small.zip'
    SMALL_FILENAME = 'fma_small.zip'

    def downloadAndExtract(self):
        self._download(self.SMALL_LINK, self.SMALL_FILENAME)
        self._extractZip(self.SMALL_FILENAME)
        self._deleteCompressedFile(self.SMALL_FILENAME)
