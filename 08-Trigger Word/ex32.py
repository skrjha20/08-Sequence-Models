import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import matplotlib.pyplot as plt
from td_utils import *

def get_random_time_segment(segment_ms):
    segment_start = np.random.randint(low=0, high=10000 - segment_ms)
    segment_end = segment_start + segment_ms - 1
    return (segment_start, segment_end)

def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time
    overlap = False
    for previous_start, previous_end in previous_segments:
        if not (segment_start > previous_end or segment_end < previous_start):
            overlap = True
    return overlap

def insert_audio_clip(background, audio_clip, previous_segments):
    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)

    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)
    previous_segments.append(segment_time)
    new_background = background.overlay(audio_clip, position=segment_time[0])

    return new_background, segment_time

def insert_ones(y, segment_end_ms):
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < y.shape[1]:
            y[0, i] = 1
    return y

def create_training_example(background, activates, negatives):
    np.random.seed(18)
    background = background - 20
    y = np.zeros((1, Ty))
    previous_segments = []

    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    for random_activate in random_activates:
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end)

    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    for random_negative in random_negatives:
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    background = match_target_amplitude(background, -20.0)

    file_handle = background.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")

    x = graph_spectrogram("train.wav")
    return x, y

if __name__ == "__main__":

    plt.figure(1)
    x = graph_spectrogram("audio_examples/example_train.wav")
    plt.show()
    _, data = wavfile.read("audio_examples/example_train.wav")
    print("Time steps in audio recording before spectrogram", data[:, 0].shape)
    print("Time steps in input after spectrogram", x.shape)

    Tx = 5511
    n_freq = 101
    Ty = 1375

    activates, negatives, backgrounds = load_raw_audio()

    print("background len: " + str(len(backgrounds[0])))
    print("activate[0] len: " + str(len(activates[0])))
    print("activate[1] len: " + str(len(activates[1])))

    overlap1 = is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
    overlap2 = is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
    print("Overlap 1 = ", overlap1)
    print("Overlap 2 = ", overlap2)

    arr1 = insert_ones(np.zeros((1, Ty)), 9700)
    plt.figure(2)
    plt.plot(insert_ones(arr1, 4251)[0, :])
    print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])
    plt.show()

    x, y = create_training_example(backgrounds[0], activates, negatives)
    plt.figure(3)
    plt.plot(y[0])
    plt.show()

