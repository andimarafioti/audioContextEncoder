# Audio inpainting with a context encoder

This project accompanies the research work on audio inpainting of small gaps done at the Acoustics Research Institute in Vienna collaborating with the Swiss Data Science Center.

# Installation

Install the requirements with `pip install -r requirements.txt`. For windows users, the numpy version should be 1.14.0+mkl (find it [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/)). For the FMA dataset, librosa requires ffmpeg as an mp3 backend. 

# Instructions
The paper uses both google's Nsynth dataset and the FMA dataset. In order to recreate the used dataset, execute in the parent folder either `python make_nsynthdataset.py` or  `python make_fmadataset.py`. The output of the scripts are three `tfrecord` files for training, validating and testing the model.
 
The default parameters for the network come pickled in the file `Papers_Context_Encoder_parameters.pkl`. In order 
to make other architectures use [saveParameters.py](utils/saveParameters.py).
 
To train the network, execute in the parent folder `python paperArchitecture.py`. This will train the network for 600k steps with a learning rate of 1e-3. You can select on which tfrecords to train the network, the script assumes you have created the nsynth dataset.
