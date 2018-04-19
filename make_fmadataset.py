import librosa
from audioread import NoBackendError

from datasetGenerator.exampleProcessor import ExampleProcessor
from datasetGenerator.fmaDownloader import FMADownloader
from datasetGenerator.fmaTFRecordGenerator import FMATFRecordGenerator

__author__ = 'Andres'

try:  # Test the backend for mp3 files
    librosa.load("utils/test/098569.mp3")
except NoBackendError as e:
    raise e

downloader = FMADownloader()
downloader.downloadAndExtract()

exampleProcessor = ExampleProcessor(gapLength=1024, sideLength=2048, hopSize=512, gapMinRMS=1e-3)

tfRecordGenerator = FMATFRecordGenerator(baseName='FMA-test', pathToDataFolder=downloader.TEST_DIR, exampleProcessor=exampleProcessor)
tfRecordGenerator.generateDataset()

tfRecordGenerator = FMATFRecordGenerator(baseName='FMA-valid', pathToDataFolder=downloader.VALID_DIR, exampleProcessor=exampleProcessor)
tfRecordGenerator.generateDataset()

tfRecordGenerator = FMATFRecordGenerator(baseName='FMA-train', pathToDataFolder=downloader.TRAIN_DIR, exampleProcessor=exampleProcessor)
tfRecordGenerator.generateDataset()
