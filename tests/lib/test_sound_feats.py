import unittest
from lib import training
import os
basedir = os.path.dirname(os.path.abspath(__file__))

class TestLib(unittest.TestCase):
    def setUp(self):
        # Data from https://freewavesamples.com/roland-jv-2080-pick-bass-c2
        self.wave_filename = os.path.join(basedir,"../../data/audio_train/0a0a8d4c.wav")
        self.wave_dir = os.path.join(basedir, "../../data/audio_train/")
        self.train_csv_file = os.path.join(basedir,"../../data/text/train.csv")
        pass
    def test_wave_info(self):
        _training = training.Training(self.train_csv_file,base_data_path=self.wave_dir, subset_ratio=0.01)
        print(_training.train_mode())


