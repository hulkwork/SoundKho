import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
import numpy as np


basedir = os.path.dirname(os.path.abspath(__file__))


class WaveFeatures(object):
    def __init__(self, wave_file, sampling_rate=16000, audio_duration=0.5, plot=True):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.wave_file_name = wave_file
        self.plot = plot
        self.read()

    def read(self):
        self.rate, self.data = wavfile.read(self.wave_file_name)

    def get_feat(self):
        self.data_norm = audio_norm(self.data)
        self.data_norm_duration = self.data_norm[:int(self.sampling_rate * self.audio_duration)]
        self.fft_data = np.abs(np.fft.rfft(self.data_norm_duration))
        if self.plot:
            yf = self.fft_data
            fig, ax = plt.subplots()
            ax.plot(range(len(yf)), yf)

            plt.show()
        return self.fft_data


def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 1e-6)
    return data - 0.5
