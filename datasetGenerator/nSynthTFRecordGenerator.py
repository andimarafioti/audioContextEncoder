from datasetGenerator.tfRecordGenerator import TFRecordGenerator

__author__ = 'Andres'


class NSynthTFRecordGenerator(TFRecordGenerator):
    def _filenameShouldBeLoaded(self, filename):
        return filename.endswith('.wav')


if __name__ == "__main__":
    from datasetGenerator.exampleProcessor import ExampleProcessor

    exampleProcessor = ExampleProcessor()
    tfRecordGen = NSynthTFRecordGenerator(baseName='test', pathToDataFolder='nsynth-test/audio', exampleProcessor=exampleProcessor)
    tfRecordGen.generateDataset()


