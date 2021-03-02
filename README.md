# AhemPreventor - a subconscious training system to prevent filler words
The AhemPreventor is a system that detects filler words in oral presentations through a neural network. It discretely gives presenters tactile feedback whenever they use a filler word. Presenters may use this system to train themselves to use fewer filler words in presentations subconsciously.

## The Idea
The initial idea for the AhemPreventor was born from learning presentation techniques ourselves. One way to avoid using filler words is to place a trusted person in the audience who will signify the use of filler words through a previously agreed gesture (e.g., scratching his head). The presenter can then consciously remind him- or herself not to use this word anymore. Of course, this takes training and time and having a trusted person in the audience is not always possible.

Thus, the idea for the AhemPreventor emerged: use machine learning to recognize one or more filler words and then give discrete tactile feedback every time a presenter uses a filler word. This method does not require another person in the audience. The use of tactile feedback (e.g., on the wrist through a smartwatch) leads to a somewhat subconscious training effect as the presenter may also ignore the tactile feedback in stressful situations.


## Why we decided to release this project as open-source
The AhemPreventor is a research project conducted by the Human-Computer Interaction Group at the Leibniz University Hannover. Since the main author's Ph.D. topic was not related to this project, the AhemPreventor is not officially published as a conference paper yet. Go ahead if you like the idea! 


## Data Collection
We gathered data from our presentations to students at the university in a "Programming 1" lecture and exercise. In total, we collected 14 hours of audio data from three different presenters in 12 presentations. This raw data was then manually labeled by specifying the "ahem" filler word's start and end.

The raw data cannot be released to the public due to privacy concerns of the presenters. However, we still release our scripts for extracting features from the raw data alongside the raw extracted features (e.g., fft, mfcc, fbank) as numpy data files.


## Adventures into Machine Learning
We extracted various possibly interesting features from the raw audio, including fft, mfcc, and fbank features. These features were then fed into various neural network architectures to find an architecture that performs well enough for the system's productive use. We were mainly looking to keep the rate of false positives as low as possible as we noticed false positives to have a strong negative impact on the perception of the system during our initial tests.

## Preliminary Results
Below, we present some preliminary results on the accuracy of our system.



## Limitations
We all used the same microphone for recording our presentations. Thus the data is hardly generalizable. We dealt with this limitation by manually adding gaussian noise to the data and generating more samples from the available audio. 

To build a productive system, future projects should collect a lot more data from various microphones and speakers. Even then, a per-user calibration or rather an individual user training is advised, as the use and type of filler words differ significantly between different people.


## Required Python Libraries to run the Project Jupyter Notebook Files
Apart from working Python 3, Tensorflow, and Keras installations, you'll need the following python libraries:

pydub, python_speech_features, simpleaudio

You can install these libraries with the following command:
`python -m pip install libraryName`



## Contact
* Oliver Beren Kaul - beren@kaul.me
* Michael Rohs - [michael.rohs@hci.uni-hannover.de](mailto:michael.rohs@hci.uni-hannover.de "michael.rohs@hci.uni-hannover.de")
