from datasetGenerator.tfRecordGenerator import TFRecordGenerator

__author__ = 'Andres'


class FMATFRecordGenerator(TFRecordGenerator):
    def _filenameShouldBeLoaded(self, filename):
        return filename.endswith('.mp3')
