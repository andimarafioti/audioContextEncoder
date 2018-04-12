from datasetGenerator.exampleProcessor import ExampleProcessor
from datasetGenerator.fmaDownloader import FMADownloader
from datasetGenerator.fmaTFRecordGenerator import FMATFRecordGenerator

__author__ = 'Andres'

baseName = 'test'
pathToDataFolder = 'FMA-test'

downloader = FMADownloader()
downloader.downloadAndExtract()

exampleProcessor = ExampleProcessor(gapLength=1024, sideLength=2048, hopSize=512, gapMinRMS=1e-3)
tfRecordGenerator = FMATFRecordGenerator(baseName, pathToDataFolder, exampleProcessor)
tfRecordGenerator.generateDataset()
