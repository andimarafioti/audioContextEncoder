from datasetGenerator import tfRecordGenerator
#generator =  tfRecordGenerator.TFRecordGenerator(baseName='/store/nati/datasets/Nsynth/valid', pathToDataFolder='/store/nati/datasets/Nsynth/nsynth-valid/audio', windowSize=5120, gapLength=1024, hopSize=512, notifyEvery=100000)

generator =  tfRecordGenerator.TFRecordGenerator(baseName='/store/nati/datasets/Nsynth/train', pathToDataFolder='/store/nati/datasets/Nsynth/nsynth-train/audio', windowSize=5120, gapLength=1024, hopSize=512, notifyEvery=100000)

generator.generateDataset()
