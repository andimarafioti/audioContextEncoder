import ssl
import urllib.request
import tarfile
import zipfile
import os

__author__ = 'Andres'


class Downloader(object):
    def downloadAndExtract(self):
        for link, filename in self._downloadLinksAndFilenames():
            self._download(link, filename)
            self._extractCompressedFile(filename)
            self._deleteCompressedFile(filename)
        self._divideDataIntoTrainValidAndTestSubsets()

    def _download(self, aLink, toAFilename):
        print("Downloading to ", toAFilename)
        size = 0
        blocksize = 4096

        with urllib.request.urlopen(aLink, context=ssl.SSLContext(ssl.PROTOCOL_TLSv1)) as response, \
                open(toAFilename, 'wb') as out_file:  # context avoids SSL certifications
            length = float(response.getheader('content-length'))
            data = response.read(blocksize)
            out_file.write(data)
            while data:
                size += len(data)
                print('\r Downloaded {:.2f} % '.format(100 * size / length), end='')
                data = response.read(blocksize)
                out_file.write(data)
            print('')

    def _deleteCompressedFile(self, filename):
        print('Deleting', filename)
        os.remove(filename)

    def _extractTar(self, aFile):
        print('Extracting', aFile)
        tar = tarfile.open(aFile)
        tar.extractall()
        tar.close()

    def _extractZip(self, aFile):
        print('Extracting', aFile)
        zip_ref = zipfile.ZipFile(aFile, 'r')
        zip_ref.extractall()
        zip_ref.close()

    def _extractCompressedFile(self, filename):
        raise NotImplementedError("Subclass Responsibility")

    def _downloadLinksAndFilenames(self):
        raise NotImplementedError("Subclass Responsibility")

    def _divideDataIntoTrainValidAndTestSubsets(self):
        raise NotImplementedError("Subclass Responsibility")
