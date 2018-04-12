from datasetGenerator.exampleProcessor import ExampleProcessor
from datasetGenerator.nSynthDownloader import NSynthDownloader
from datasetGenerator.nSynthTFRecordGenerator import NSynthTFRecordGenerator

__author__ = 'Andres'


downloader = NSynthDownloader()
downloader.downloadAndExtract()

exampleProcessor = ExampleProcessor(gapLength=1024, sideLength=2048, hopSize=512, gapMinRMS=1e-3)

tfRecordGenerator = NSynthTFRecordGenerator(baseName='nsynth_test', pathToDataFolder=downloader.TEST_DIR, exampleProcessor=exampleProcessor)
tfRecordGenerator.generateDataset()

tfRecordGenerator = NSynthTFRecordGenerator(baseName='nsynth_valid', pathToDataFolder=downloader.VALID_DIR, exampleProcessor=exampleProcessor)
tfRecordGenerator.generateDataset()

tfRecordGenerator = NSynthTFRecordGenerator(baseName='nsynth_train', pathToDataFolder=downloader.TRAIN_DIR, exampleProcessor=exampleProcessor)
tfRecordGenerator.generateDataset()
