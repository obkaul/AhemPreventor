# AhemPreventor - a subconscious training system to prevent filler words
The AhemPreventor is a system that detects filler words in oral presentations through a neural network. It discretely gives presenters tactile feedback whenever they use a filler word. Presenters may use this system to train themselves to use fewer filler words in presentations subconsciously.

## The Idea
The initial idea for the AhemPreventor was born from learning presentation techniques ourselves. One way to avoid using filler words is to place a trusted person in the audience who will signify the use of filler words through a previously agreed gesture (e.g., scratching his head). The presenter can then consciously remind him- or herself not to use this word anymore. Of course, this takes training and time and having a trusted person in the audience is not always possible.

Thus, the idea for the AhemPreventor emerged: use machine learning to recognize one or more filler words and then give discrete tactile feedback every time a presenter uses a filler word. This method does not require another person in the audience. The use of tactile feedback (e.g., on the wrist through a smartwatch) leads to a somewhat subconscious training effect as the presenter may also ignore the tactile feedback in stressful situations.


## Data Collection
We gathered data from our presentations to students at the university in a "Programming 1" lecture and exercise. In total, we collected 14 hours of audio data from three different presenters in 12 presentations. This raw data was then manually labeled by specifying the "ahem" filler word's start and end.


## Adventures in Machine Learning
We extracted various possibly interesting features from the raw audio, including spectrograms, mfcc, and fbank features. These features were then fed into various neural network architectures to find an architecture that performs well enough for the system's productive use. We were mainly looking to keep the rate of false positives as low as possible as we noticed false positives to have a strong negative impact on the perception of the system during our initial tests.

As we had much fewer positive than negative samples, we generated additional positive samples by applying a random amount of Gaussian noise to all samples. Here, it is essential to (a) apply the noise to all samples, positive and negative and (b) pick the validation set carefully so that no original positive samples with different noise levels are added to both, training and validation set. When disregarding case (a), we would just build a noise detector, and disregarding case (b) would train validation samples with slightly different noise values, which is certainly undesirable.

## Preliminary Results
Below, we present some preliminary results on the accuracy of our system.

![Result highlights](https://github.com/obkaul/AhemPreventor/blob/main/preliminaryResults/highlights.png)


## Limitations
We all used the same microphone for recording our presentations. Thus the data is hardly generalizable. We mitigated this limitation by manually adding Gaussian noise to multiples of the data and generating extra samples. 

Future projects should collect a lot more data from various microphones and speakers to build a productive system. Even then, a per-user calibration or rather an individual user training is advised, as the use and type of filler words differ significantly between different people.


## Why we decided to release this project as open-source
The AhemPreventor is a research project conducted by the Human-Computer Interaction Group at the Leibniz University Hannover. Since the main author's Ph.D. topic was not related to this project, the AhemPreventor is not officially published as a conference paper yet. Go ahead if you like the idea! Otherwise, this project might also serve as a learning experience in audio classification for others, as we learned a lot while implementing these architectures.

### What is included in this release
* A helper library to define common functions and keep a few global variables constant over all scripts,
* our jupyter notebooks for extracting features from the labeled .wav files,
* part of one of the files we extracted data from: AudioData/Uebung-02-reencoded.wav and all labeling files,
* extracted mfcc features as .npy files,
* some baseline and some promising neural network architectures (We tested a lot more, but we discarded those that did not yield an acceptable performance except for one simple dense network for comparison,
* The generated model.h5 files

### What is not included:
* The raw wav data of most presentations cannot be released to the public due to some of the presenters' privacy concerns. We release part of one of the files as a sample. We could not upload the entire file due to Githubs 100 MB file size limit.
* We will also not release the tactile system component, which uses one of the trained neural networks to give tactile feedback while listening to a microphone input. We believe this is of little interest to the public / AI community as we used a custom tactile wristband. It would be easier and more accessible to write a small add-on to actuate a smartwatch. In case you'd like access to the tactile component anyway, please shoot us an email!

### How the mfcc features were calculated (see library.py):

    # sampleData is an array of 750 ms of audio data at sampling_frequency=16 Khz.
    # limitLowFreq and limitHighFreq define a window of 64..2800 Hz to capture the most important features from human speech.
    # The mfcc function originates from the python_speech_features library.
    
    mfccResult = mfcc(sampleData, samplerate=sampling_frequency, winlen=0.03, winstep=0.015, numcep=40,
                    nfilt=40, nfft=512, lowfreq=limitLowFreq, highfreq=limitHighFreq, preemph=0.97,
                    ceplifter=22, appendEnergy=True)
        
    # normalize
    mfccResult -= (np.mean(mfccResult, axis=0) + 1e-8)

The resulting mfcc features with a shape of (None, 49, 40) are fed directly into the model.fit and model.predict functions of our various neural network approaches.

We also tried feeding raw data, spectrograms, fbank and mfcc and fbank combined features into the network architectures. However, the mfcc features yielded the best results in our case.

## Required Python Libraries to run the Project Jupyter Notebook Files
Apart from working Python 3, Tensorflow, and Keras installations, you'll need the following python libraries:

numpy, matplotlib, pydub, python_speech_features, simpleaudio

You can install these libraries with the following command:
`python -m pip install libraryName`



## Contact
* Oliver Beren Kaul - beren@kaul.me
* Michael Rohs - [michael.rohs@hci.uni-hannover.de](mailto:michael.rohs@hci.uni-hannover.de "michael.rohs@hci.uni-hannover.de")
