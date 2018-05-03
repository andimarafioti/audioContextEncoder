from datasetGenerator.exampleProcessor import ExampleProcessor
from datasetGenerator.fakeTFRecordGenerator import FakeTFRecordGenerator

__author__ = 'Andres'


exampleProcessor = ExampleProcessor(gapLength=1024, sideLength=2048, hopSize=512, gapMinRMS=1e-3)

tfRecordGenerator = FakeTFRecordGenerator(baseName='fake', pathToDataFolder='', exampleProcessor=exampleProcessor)
tfRecordGenerator.generateDataset()
