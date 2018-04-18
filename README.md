# Audio inpainting with a context encoder

This folder accompanies the research work on audio inpainting of small gaps done at the Acoustics Research Institute in Vienna collaborating with the Swiss Data Science Center.

The paper uses both google's Nsynth dataset and the FMA dataset. In order to recreate the used dataset, execute in the parent folder either `python make_nsynthdataset.py` or  `python make_fmadataset.py` depending on which dataset you need. The output of the scripts are three `tfrecord` for training, validating and testing the model.
 
To train the network, execute in the parent folder `python3 paperArchitecture.py`. This will train the network for 600k steps with a learning rate of 1e-3. You can select on which tfrecords to train the network, the script assumes you have created the nsynth dataset.
