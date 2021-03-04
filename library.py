"""

AhemPreventor helper library

Authors: Oliver Beren Kaul and Michael Rohs

Developed in 2018 - 2021

"""


import sys
import time

# http://www.numpy.org
import numpy as np

# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html
import matplotlib.pyplot as pp

# Multidimensional matrix multiplications
import tensorflow as tf

# Convenient API for describing neural networks
import keras as k
from keras.utils import np_utils

# Reading/writing datasets from/to file 
import h5py

# https://github.com/jiaaro/pydub
from pydub import AudioSegment
from pydub.playback import play

import random

from python_speech_features import mfcc, fbank, logfbank


def set_global_params():
    global duration, preContext, sampling_frequency, samples, window_size, step_size, windows, limitLowFreq, limitHighFreq, positivesMultiplier
    # 95% percentile positives duration: 500 ms + 300 ms context
    duration = 600 # ms, duration of frame
    preContext = 150 # include x ms before the start of the sample
    sampling_frequency = 16000 # samples / sec
    samples = duration * sampling_frequency // 1000 # samples per frame
    window_size = 512 # window size of fft samples
    step_size = 128 # step size of fft samples
    windows = (samples - window_size) // 256 # number of windows
    limitLowFreq = 64  #  most features of human speech exist between these boundaries
    limitHighFreq = 2800  #  most features of human speech exist between these boundaries
    positivesMultiplier = 25  #  since we have a lot more negative than positive samples, we should augment our positive samples and add them multiple times
    
    print("AhemPreventor library load success.")
set_global_params()

def load_audio(filename):
    global sampling_frequency
    a = AudioSegment.from_file(filename)
    a = a.set_frame_rate(sampling_frequency)
    a.set_channels(1)
    return a


def getSample(wavefile, start_pos, augmentWithNoise = False):
    """get a single sample with a defined start_pos and with extra preContext from the audio file, possibly augment it with noise"""
    if start_pos - preContext < 0 or start_pos + duration > len(wavefile):
        return None
    
    currentSample = wavefile[start_pos - preContext : start_pos + duration]
    currentSampleArray = np.array(currentSample.get_array_of_samples())
    
    if augmentWithNoise:
        currentSampleArray = augmentAudioSampleWithNoise(currentSampleArray)
    
    return currentSampleArray
    
def augmentAudioSampleWithNoise(audioSample):
    """Augment the audio sample with gaussian noise no stronger than maxGaussianNoiseMagnitude"""
    maxGaussianNoiseMagnitude = 0.3 
    # mean and standard deviation of noise
    mu = 1 
    sigma = np.random.uniform(0, random.uniform(0.0, maxGaussianNoiseMagnitude)) 
    
    randomNoise = np.random.normal(mu, sigma, len(audioSample)) # generate random noise based on the random sigma..
    
    audioSample = audioSample * randomNoise   
    
    return audioSample
    
def overlaps_position(start, end, positions, positions_negative):
    """Checks if the interval [start, end] overlaps with any of the positive sample positions. Also check, wehther they overlap with an already existing negative sample"""
    for pstart, pend in positions:
        if end >= pstart and start <= pend:
            return True
    for pstart, pend in positions_negative:
        if end >= pstart and start <= pend:
            return True
    return False
    

def compute_rawData(sampleData):
    """Just normalize and shape the raw data"""
    sampleData = normalizeArray(sampleData)
    
    # simply reshape our data such that we have multiple windows of the raw data, might help using recurrent networks
    return np.reshape(sampleData, (60,60))

def compute_spectrum(sampleData, mid_pos):
    """Computes spectrum in AudioSegment a around (+/-0.5*duration) mid_pos."""
    overlap = window_size - step_size # overlap between windows
    spectrum, freqs, times, img = pp.specgram(
        sampleData, window_size, sampling_frequency, noverlap = overlap)
    # print("spectrum =", spectrum.shape)
    n_freq_bins = 65 # cut at frequency, frequency bin 65 corresponds to roughly 2800 Hz
    return spectrum[:n_freq_bins,:]

def compute_mfcc(sampleData):   
    """calculate mfcc features, normalize the output"""    
    mfccResult = mfcc(sampleData, samplerate=sampling_frequency, winlen=0.03, winstep=0.015, numcep=40,
                      nfilt=40, nfft=512, lowfreq=limitLowFreq, highfreq=limitHighFreq, preemph=0.97,
                      ceplifter=22, appendEnergy=True)
    
    # normalize
    mfccResult -= (np.mean(mfccResult, axis=0) + 1e-8)
    return mfccResult
    
def compute_fbank(sampleData):
    """calculate fbank features, normalize the output"""
    fbankResult = fbank(sampleData, samplerate=sampling_frequency, winlen=0.035, winstep=0.01,
                      nfilt=40, nfft=512, lowfreq=limitLowFreq, highfreq=limitHighFreq, preemph=0.97, winfunc=np.hamming)

    fbankResult = fbankResult[0]
    
    #normalize
    fbankResult -= (np.mean(fbankResult, axis=0) + 1e-8)
    return fbankResult
    
def compute_mfccAndFbank(sampleData):
    """calculate mfcc and fbank features, normalize the output, concatenate them"""
    mfccFeatures = compute_mfcc(sampleData)
    fbankFeatures = compute_fbank(sampleData)    
    return np.concatenate((mfccResult, fbankResult), axis=1)

def log_normalize_spectrum(s, mean = None, std = None):
    """Log-transforms and normalizes data (zero mean, unit variance)."""
    s = np.log(s + 1e-10) # add 1e-10 to avoid -inf
    if mean is None:
        mean = np.mean(s)
    if std is None:
        std = np.std(s)
    s = (s - mean) / std
    return s

def normalize_training_data(x_train):
    """Log-transforms and normalizes data (zero mean, unit variance)."""
    x_train = np.log(x_train + 1e-10) # add 1e-10 to avoid -inf
    print(np.isnan(x_train).any())
    mean = np.mean(x_train)
    std = np.std(x_train)
    print(mean, std)
    x_train = (x_train - mean) / std
    return x_train

# https://en.wikipedia.org/wiki/Sensitivity_and_specificity
# https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def normalizeArray(arr):
    return minmax_scale(arr.astype(np.float32))

def false_positive_rate(y_true, y_pred, threshold = 0.5):
    y_true = tf.cast(y_true > threshold, tf.int32)
    y_pred = tf.cast(y_pred > threshold, tf.int32)
    TP_FP = tf.cast(tf.reduce_sum(y_pred), tf.float32)
    # TP = tf.reduce_sum(y_pred * y_true)
    FP = tf.cast(tf.reduce_sum(y_pred * (1 - y_true)), tf.float32)
    return tf.divide(FP, (tf.add(TP_FP, tf.constant(1e-10)))) # FP / (TP + FP)

# TP / P, recall
def true_positive_rate(y_true, y_pred, threshold = 0.5):
    y_true = tf.cast(tf.math.greater(y_true, threshold), tf.int32)
    y_pred = tf.cast(tf.math.greater(y_pred, threshold), tf.int32)
    P = tf.cast(tf.reduce_sum(y_true), tf.float32) # number of real positive cases
    TP = tf.cast(tf.reduce_sum(y_pred * y_true), tf.float32) # number of correctly identified positive cases
    return tf.divide(TP, tf.add(P, tf.constant(1e-10)))


def show_history(history, class_weight = None):
    pp.figure()
    pp.plot(history.history['accuracy'], label="accuracy")
    pp.plot(history.history['val_accuracy'], label="val_accuracy")
    pp.xlabel("epochs")
    pp.ylabel("accuracy")
    pp.grid()
    pp.legend()

    pp.figure()
    pp.plot(history.history['false_positive_rate'], label="false_positives")
    pp.plot(history.history['val_false_positive_rate'], label="val_false_positives")
    pp.xlabel("epochs")
    pp.ylabel("false positive rate")
    pp.grid()
    pp.legend()
    if not class_weight is None:
        pp.title("class_weight=" + str(class_weight))
    
    pp.figure()
    pp.plot(history.history['true_positive_rate'], label="true_positives")
    pp.plot(history.history['val_true_positive_rate'], label="val_true_positives")
    pp.xlabel("epochs")
    pp.ylabel("true positive rate")
    pp.grid()
    pp.legend()
    if not class_weight is None:
        pp.title("class_weight=" + str(class_weight))

    
def show_histories(history1, history2):
    pp.figure()
    pp.plot(history1.history['accuracy'], label="1_acc")
    pp.plot(history1.history['val_accuracy'], label="1_val_acc")
    pp.plot(history2.history['accuracy'], label="2_acc")
    pp.plot(history2.history['val_accuracy'], label="2_val_acc")
    pp.xlabel("epochs")
    pp.ylabel("accuracy")
    pp.grid()
    pp.legend()