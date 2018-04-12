from datasetGenerator.exampleProcessor import ExampleProcessor
from datasetGenerator.nSynthDownloader import NSynthDownloader
from datasetGenerator.nSynthTFRecordGenerator import NSynthTFRecordGenerator

__author__ = 'Andres'


downloader = NSynthDownloader()
downloader.downloadAndExtract()

exampleProcessor = ExampleProcessor(gapLength=1024, sideLength=2048, hopSize=512, gapMinRMS=1e-3)

tfRecordGenerator = NSynthTFRecordGenerator(baseName='test', pathToDataFolder='nsynth-test/audio', exampleProcessor=exampleProcessor)
tfRecordGenerator.generateDataset()

tfRecordGenerator = NSynthTFRecordGenerator(baseName='valid', pathToDataFolder='nsynth-valid/audio', exampleProcessor=exampleProcessor)
tfRecordGenerator.generateDataset()

tfRecordGenerator = NSynthTFRecordGenerator(baseName='train', pathToDataFolder='nsynth-train/audio', exampleProcessor=exampleProcessor)
tfRecordGenerator.generateDataset()
