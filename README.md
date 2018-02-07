# Audio inpainting with a context encoder

This folder accompanies the research work on audio inpainting of small gaps done at the Acoustics Research Institute in Vienna collaborating with the Swiss Data Science Center.

The paper uses google's Nsynth dataset as a training set. In order to recreate the used data, one needs to instantiate the TFRecordGenerator class with baseName='paper_recreation', pathToDataFolder='path/to/NSynth', window_size=5192, gapLength=1024, hopSize=512.
 
 With that dataset one can train the network following the 'train' notebook. 
 You can also use the 'test' notebook to see the results presented in the paper.
 
 Finally, with the 'try' notebook one can try training the network with diferent parameters.
